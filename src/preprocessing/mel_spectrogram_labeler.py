import os
import pandas as pd
from pathlib import Path
from preprocessing.file_system_helper import FileSystemHelper

class MelSpectrogramLabeler:
    def __init__(self, base_dir):
        """
        Class for labeling Mel spectrogram files within a directory structure.

        Args:
            base_dir (str or Path): Base directory path containing Mel spectrogram data.

        Attributes:
            base_dir (Path): Path to the base directory containing Mel spectrogram data.
            data (list of dict): List to store labeled spectrogram data in the format {'file_path': str, 'label': str}.
        """
        self.base_dir = Path(base_dir)
        self.data = []

    def label_spectrogram(self, mel_dir_name, extension):
        """
        Label Mel spectrogram files within artist directories in the base directory.

        Args:
            mel_dir_name (str): Name of the directory containing Mel spectrogram files.
            extension (str): File extension of Mel spectrogram files.

        Returns:
            None
        """
        for artist_dir in self.base_dir.iterdir():
            if not artist_dir.is_dir():
                continue

            artist_name = artist_dir.name
            mel_dir = artist_dir.joinpath(mel_dir_name)

            if mel_dir.exists() and mel_dir.is_dir():
                mel_files = FileSystemHelper.get_files_by_extension(mel_dir, extension)
                for mel_file in mel_files:
                    self.data.append({
                        'file_path': str(mel_file),
                        'label': artist_name
                    })

    def save_to_dataframe(self):
        """
        Convert labeled spectrogram data to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing 'file_path' and 'label' columns.
        """
        return pd.DataFrame(self.data)

    def save_to_csv(self, output_path):
        """
        Save labeled spectrogram data to a CSV file.

        Args:
            output_path (str or Path): Path to save the CSV file.

        Returns:
            None
        """
        df = self.save_to_dataframe()
        df.to_csv(output_path, index=False)

