import sounddevice as sd
import numpy as np
import openwakeword
from openwakeword.model import Model
import threading
import queue
import time

#Constants
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280

class Ears:
    def __init__(self, sensitivity: float = 0.5):
        '''
        Implementation of the threaded microphone listener.
        Args:
            sensitivity: The sensitivity of the wake word detector.
        '''

        self.sensitivity = sensitivity
        self.listening = False

        #The buffer
        self.audio_queue = queue.Queue()

        #The model
        print("Loading the model...")

        # Work around the installed openwakeword version where AudioFeatures.__init__
        # does not accept the `inference_framework` keyword that Model passes.
        try:
            import openwakeword.utils as oww_utils

            AudioFeatures = oww_utils.AudioFeatures
            original_init = AudioFeatures.__init__

            # Only patch if the current signature doesn't already accept the kwarg
            if hasattr(original_init, "__code__") and "inference_framework" not in original_init.__code__.co_varnames:

                def _patched_audiofeatures_init(self, *args, **kwargs):
                    # Drop unsupported kwarg and delegate to original initializer
                    kwargs.pop("inference_framework", None)
                    return original_init(self, *args, **kwargs)

                AudioFeatures.__init__ = _patched_audiofeatures_init  # type: ignore[assignment]
        except Exception as e:
            print(f"Warning: could not apply openwakeword AudioFeatures compatibility patch: {e}")

        # Instantiate the wake word model (will lazily download resources if needed)
        self.model = Model(inference_framework="onnx")
        print("Model loaded successfully.")

        # Cooldown timestamp (monotonic seconds) to ignore audio after a wake event
        self._cooldown_until = 0.0

    def _mic_callback(self, indata, frames, time, status):
        '''
        Callback function for the microphone.
        '''

        if status:
            print(f"Microphone status: {status}")

        self.audio_queue.put(indata.copy())

    def _consumer_loop(self, on_wake):
        '''
        This runs in the background thread and processes the audio data.
        '''

        print('Ears: starting listening...')

        #Opening microphone stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=CHUNK_SIZE,
            dtype="int16",
            channels=1,
            latency="high",  # Prefer higher latency to reduce buffer underruns
            callback=self._mic_callback,
        ):

            while self.listening:
                #Get audio data from the queue
                chunk = self.audio_queue.get()

                # Ensure audio is 1D int16 as expected by openwakeword's AudioFeatures
                if isinstance(chunk, np.ndarray):
                    if chunk.ndim == 2:
                        # Flatten mono channel (frames, 1) -> (frames,)
                        chunk = chunk[:, 0]
                    elif chunk.ndim > 2:
                        chunk = chunk.reshape(-1)
                    chunk = chunk.astype(np.int16, copy=False)

                # Respect cooldown window: keep draining queue but skip inference
                now = time.monotonic()
                if now < self._cooldown_until:
                    continue

                #Process the audio data
                predictions = self.model.predict(chunk)

                #Check if the wake word is detected
                for model_name, score in predictions.items():
                    if score > self.sensitivity:
                        print(f"Wake word detected: {model_name} with score {score}")

                        # This callback may block for several seconds while recording/transcribing.
                        # During this time the producer keeps filling `audio_queue`.
                        on_wake()

                        # Immediately flush any audio that arrived during the wake handling
                        try:
                            while True:
                                self.audio_queue.get_nowait()
                        except queue.Empty:
                            pass

                        # Reset model state and start a short cooldown where we ignore new audio
                        self.model.reset()
                        self._cooldown_until = time.monotonic() + 2.0

                        # Only handle a single wake trigger per chunk
                        break

    def start(self, on_wake_callback):
        '''
        Start the listener in a seperate thread.
        '''
        self.listening = True

        #Creating a new thread for the consumer loop
        self.thread = threading.Thread(
            target=self._consumer_loop,
            args=(on_wake_callback,),
            daemon=True,
        )

        #Starting the thread
        self.thread.start()

    def stop(self):
        '''
        Stop the listener.
        '''
        self.listening = False
        if hasattr(self, 'thread'):
            self.thread.join()

#Testing
if __name__ == "__main__":
    import time
    
    def wake_up_action():
        print("Wake up action!")

    ears = Ears(sensitivity=0.5)
    
    try:
        ears.start(on_wake_callback=wake_up_action)
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping ears...")
        ears.stop()

