import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

from preprocessing.file_system_helper import FileSystemHelper


class MelSpectrogramGenerator:
    def __init__(self, save_path, sr, n_mels, hop_length):
        self.save_path = save_path
        self.n_mels = n_mels
        self.sr = sr
        self.hop_length = hop_length
        FileSystemHelper.make_dir(save_path)

    def generate_and_save(self, y, file_name, hop_length):
        mel = librosa.feature.melspectrogram(y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel, sr=self.sr, hop_length=self.hop_length,  bbox_inches='tight', pad_inches=0)
        plt.close()
