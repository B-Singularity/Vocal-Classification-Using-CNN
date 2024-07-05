import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from preprocessing.mel_spectrogram_generator import MelSpectrogramGenerator
import tempfile
import os
import warnings



class TestMelSpectrogramGenerator(unittest.TestCase):

    def setUp(self):
        """Set up the test case."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.save_path = self.temp_dir.name  # Use temporary directory
        self.sr = 22050
        self.n_mels = 128
        self.hop_length = 512

        # Create a MagicMock for librosa.display.specshow
        self.mock_specshow = MagicMock()

        # Create a MagicMock for matplotlib.pyplot.savefig
        self.mock_savefig = MagicMock()

        self.generator = MelSpectrogramGenerator(self.save_path, self.sr, self.n_mels, self.hop_length)

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    @patch('librosa.feature.melspectrogram')
    @patch('librosa.power_to_db')
    @patch('librosa.display.specshow', new_callable=MagicMock)
    @patch('matplotlib.pyplot.savefig', new_callable=MagicMock)
    def test_generate_and_save(self, mock_savefig, mock_specshow, mock_power_to_db, mock_melspectrogram):
        """Test the generate_and_save method."""
        # Mock return values
        mock_mel = np.zeros((self.n_mels, 100))  # Mock Mel spectrogram
        mock_melspectrogram.return_value = mock_mel  # Ensure consistent shapes
        mock_power_to_db.return_value = mock_mel

        # Call the method under test
        test_y = np.random.randn(1000)
        test_file_name = 'test_spec'
        self.generator.generate_and_save(test_y, test_file_name)

        # Assertions to verify the behavior
        mock_melspectrogram.assert_called_once_with(test_y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
        mock_power_to_db.assert_called_once_with(mock_mel, ref=np.max)
        mock_specshow.assert_called_once()
        mock_savefig.assert_called_once_with(f"{self.save_path}/{test_file_name}.png", bbox_inches='tight',
                                             pad_inches=0)


if __name__ == '__main__':
    unittest.main()

