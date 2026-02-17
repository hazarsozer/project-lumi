import time
import sounddevice as sd
import numpy as np
from audio.ears import Ears
from audio.scribe import Scribe
from utils import play_ready_sound

def main():
    print("Starting NenOS...")
    print("Loading audio components...")
    ears = Ears(sensitivity=0.3)
    scribe = Scribe(model_size="tiny.en")
    print("Audio components loaded successfully.")

    def on_wake():
        print("Listening for command...")
        play_ready_sound() 
        
        print("🔴 REC")
        duration = 4
        fs = 16000
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()

        print("Processing audio...")
        #Flatten
        audio_flat = recording.flatten()

        #Transcribe
        text = scribe.transcribe(audio_flat)
        print(f"Transcribed text: {text}")
        print("Listening again..")

    print("Waiting for 'Hey Jarvis'..")
    try:
        ears.start(on_wake_callback=on_wake)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ears.stop()
        print("System shutdown.")

if __name__ == "__main__":
    main()