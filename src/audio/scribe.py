from faster_whisper import WhisperModel
import numpy as np
import os

class Scribe:
    def __init__(self, model_size: str = "tiny.en", device: str = "cpu", initial_prompt: str = "Lumi, Firefox, browser, desktop assistant."):
        """
        Scribe converts audio to text.
        Args:
            model_size: The size of the model to use.
            device: The device to use for inference.
            initial_prompt: Context injection for the transcription model.
        """
        self.initial_prompt = initial_prompt
        print(f"Loading Whisper model: {model_size} on {device}...")

        #Load the model on int8 quantization
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        print(f"Whisper model loaded successfully on {device}.")

    def transcribe(self, audio_data, initial_prompt: str = None):
        """
        Transcribe the audio to text.
        Args:
            audio_data: The audio array to transcribe.
            initial_prompt: Optional override for the initial prompt context.
        Returns:
            The text transcription.
        """
        #faster-whisper expects float32, but mic gives us int16
        #we normalize it to -1.0 to 1.0
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Use provided prompt or default
        prompt = initial_prompt if initial_prompt is not None else self.initial_prompt

        #Transcribe the audio
        segments, info = self.model.transcribe(audio_data, beam_size=5, initial_prompt=prompt)
        
        #Combine segments into a single text string
        text = " ".join([segment.text for segment in segments])
        return text.strip()

#Testing

if __name__ == "__main__":
    scribe = Scribe()
    print("Test complete!")