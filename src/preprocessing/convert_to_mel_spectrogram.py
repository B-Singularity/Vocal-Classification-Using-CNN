import os

from preprocessing.audio_processor import AudioProcessor
from preprocessing.file_system_helper import FileSystemHelper
from preprocessing.mel_spectrogram_generator import MelSpectrogramGenerator


class ConvertToMelSpectrogram:
    """
    A class to convert audio files in a folder to mel spectrogram images.

    Attributes:
        folder_path (str): Path to the folder containing audio files.
        save_path (str): Path to the folder where spectrogram images will be saved.
        sr (int): Sample rate for audio processing.
        duration (float): Duration of the audio clips to be processed.
        n_mels (int): Number of mel bands to generate.
        hop_length (int): Number of samples between successive frames.
        audio_processor (AudioProcessor): Instance of AudioProcessor to handle audio loading and processing.
        mel_generator (MelSpectrogramGenerator): Instance of MelSpectrogramGenerator to generate and save mel spectrograms.
    """

    def __init__(self, folder_path, save_path, sr, duration, n_mels, hop_length):
        """
        Initializes the ConverToMelSpectrogram class with provided parameters.

        Args:
            folder_path (str): Path to the folder containing audio files.
            save_path (str): Path to the folder where spectrogram images will be saved.
            sr (int): Sample rate for audio processing.
            duration (float): Duration of the audio clips to be processed.
            n_mels (int): Number of mel bands to generate.
            hop_length (int): Number of samples between successive frames.
        """
        self.folder_path = folder_path
        self.save_path = save_path
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.hop_length = hop_length

        self.audio_processor = AudioProcessor(sr, duration)
        self.mel_generator = MelSpectrogramGenerator(save_path, sr, n_mels, hop_length)

    def convert_folder_to_mel_spectrogram(self):
        """
        Converts all audio files in the specified folder to mel spectrogram images.
        The spectrogram images are saved in the specified save_path directory.
        """
        wav_files = FileSystemHelper.get_files_by_extension(self.folder_path, 'wav')
        for audio_path in wav_files:
            y, sr = self.audio_processor.load_audio(audio_path)
            audio_slices = self.audio_processor.split_audio(y)
            for i, y_slice in enumerate(audio_slices):
                file_name = f"{os.path.splitext(os.path.basename(audio_path))[0]}_part{i}.png"
                self.mel_generator.generate_and_save(y_slice, file_name)
