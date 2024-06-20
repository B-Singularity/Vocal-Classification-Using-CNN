import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

from preprocessing.file_system_helper import FileSystemHelper


class MelSpectrogramGenerator:
    """
    A class to generate and save Mel spectrograms from audio data.

    Attributes
    ----------
    save_path : str
        The path where the generated spectrograms will be saved.
    sr : int
        The sample rate of the audio data.
    n_mels : int
        Number of Mel bands to generate.
    hop_length : int
        Hop length (in samples) between successive frames.

    Methods
    -------
    generate_and_save(y, file_name):
        Generates a Mel spectrogram from audio data `y` and saves it as an image.

    """

    def __init__(self, save_path, sr, n_mels, hop_length):
        """
        Initializes the MelSpectrogramGenerator instance.

        Parameters
        ----------
        save_path : str
            The path where the generated spectrograms will be saved.
        sr : int
            The sample rate of the audio data.
        n_mels : int
            Number of Mel bands to generate.
        hop_length : int
            Hop length (in samples) between successive frames.
        """
        self.save_path = save_path
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        FileSystemHelper.ensure_directory_exists(save_path)

    def generate_and_save(self, y, file_name):
        """
        Generates a Mel spectrogram from audio data `y` and saves it as an image.

        Parameters
        ----------
        y : np.ndarray
            Audio time series.
        file_name : str
            File name to save the generated spectrogram image.
        """
        # Generate Mel spectrogram and convert to log scale
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Plot and save the spectrogram as an image
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel, sr=self.sr, hop_length=self.hop_length, cmap='viridis')
        plt.savefig(f"{self.save_path}/{file_name}", bbox_inches='tight', pad_inches=0)
        plt.close()
