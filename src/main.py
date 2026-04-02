import time
import sounddevice as sd
import numpy as np
from audio.ears import Ears
from audio.scribe import Scribe
from utils import play_ready_sound

def main():
    print("Starting Lumi...")
    print("Loading audio components...")
    ears = Ears(sensitivity=0.8)
    scribe = Scribe(model_size="tiny.en")
    print("Audio components loaded successfully.")

    def on_wake():
        print("Listening for command...")
        play_ready_sound() 
        
        # Use VAD for smart recording
        recording = ears.record_command_with_vad()
        
        if recording.size == 0:
            print("No audio recorded.")
            return

        print("Processing audio...")
        #Transcribe
        text = scribe.transcribe(recording)
        print(f"Transcribed text: {text}")
        print("Listening again..")

    print("Waiting for 'Hey Lumi'..")
    try:
        ears.start(on_wake_callback=on_wake)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ears.stop()
        print("System shutdown.")

if __name__ == "__main__":
    main()