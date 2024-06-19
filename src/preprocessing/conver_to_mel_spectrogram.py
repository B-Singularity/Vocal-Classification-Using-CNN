from preprocessing import audio_processor
from preprocessing import mel_spectrogram_generator
from preprocessing import file_system_helper
import os
class ConverToMelSpectrogram:
    def __init__(self, folder_path, save_path, sr, duration, n_mels, hop_length):
        self.folder_path = folder_path
        self.save_path = save_path
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.hop_length = hop_length

        self.audio_processor = AudioProcessor(sr, duration)
        self.mel_generator = MelSpectrogramGenerator(save_path, sr, n_mels, hop_length)

    def convert_folder_to_mel_spectrogram(self):
        wav_files = FileSystemHelper.get_wav_files(self.folder_path)
        for audio_path in wav_files:
            y, sr = self.audio_processor.load_audio(audio_path)
            audio_slices = self.audio_processor.split_audio(y)
            for i, y_slice in enumerate(audio_slices):
                file_name = f"{os.path.splitext(os.path.basename(audio_path))[0]}_part{i}.png"
                self.mel_generator.generate_mel_spectrogram(y_slice, file_name)