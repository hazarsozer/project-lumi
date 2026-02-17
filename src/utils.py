import numpy as np
import sounddevice as sd

def play_ready_sound():
    """
    Plays a futuristic 'On' tone (A4, 440Hz).
    Simple sine wave generation to avoid loading external WAV files.
    """
    fs = 44100
    duration = 0.2  # seconds
    frequency = 880 # High pitch "ping"
    
    # Generate array
    t = np.linspace(0, duration, int(fs * duration), False)
    # Simple sine wave with fade out to avoid "click"
    note = np.sin(frequency * t * 2 * np.pi)
    
    # Ensure it's float32 for sounddevice
    audio = note.astype(np.float32) * 0.5 # 0.5 volume
    
    # Play and Wait (Blocking is intentional here)
    try:
        sd.play(audio, fs)
        sd.wait()
    except Exception as e:
        print(f"⚠️ Could not play sound: {e}")