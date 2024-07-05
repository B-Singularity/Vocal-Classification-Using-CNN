import pandas as pd
import torchaudio
from torch.utils.data import Dataset


class SoundDataset(Dataset):
    """
    A dataset class for loading and processing sound files.

    Attributes:
    annotations (list): A list containing annotation data for each audio file.

    Methods:
    __len__(): Returns the total number of audio files in the dataset.
    __getitem__(index): Returns the audio signal and label for a given index.
    _get_audio_path(index): Returns the path to the audio file at the given index.
    _get_audio_label(index): Returns the label of the audio file at the given index.
    """

    def __init__(self, annotations_files):
        """
        Initializes the SoundDataset with the given annotations.

        Parameters:
        annotations_files (list): A list of annotation data where each annotation is expected
                                  to be in the format [id, path, label].
        """
        self.annotations = pd.read_csv(annotations_files)
    def __len__(self):
        """
        Returns the total number of audio files in the dataset.

        Returns:
        int: The number of audio files.
        """
        return len(self.annotations)

    def __getitem__(self, index):
        """
        Returns the audio signal and label for a given index.

        Parameters:
        index (int): The index of the audio file.

        Returns:
        tuple: A tuple containing the audio signal (torch.Tensor) and the label.
        """
        audio_path = self._get_audio_path(index)
        label = self._get_audio_label(index)
        signal, sr = torchaudio.load(audio_path)
        return signal, label

    def _get_audio_path(self, index):
        """
        Returns the path to the audio file at the given index.

        Parameters:
        index (int): The index of the audio file.

        Returns:
        str: The file path of the audio file.
        """
        path = self.annotations.iloc[index, 1]
        return path

    def _get_audio_label(self, index):
        """
        Returns the label of the audio file at the given index.

        Parameters:
        index (int): The index of the audio file.

        Returns:
        str: The label of the audio file.
        """
        return self.annotations.iloc[index, 2]

