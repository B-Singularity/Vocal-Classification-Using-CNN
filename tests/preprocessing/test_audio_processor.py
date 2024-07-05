import unittest
from unittest.mock import patch

import numpy as np

from preprocessing.audio_processor import AudioProcessor


class TestAudioProcessor(unittest.TestCase):

    def setUp(self):
        self.sr = 22050
        self.duration = 2.0
        self.processor = AudioProcessor(self.sr, self.duration)

        t = np.linspace(0, self.duration, int(self.sr * self.duration), endpoint=False)
        self.test_tone = 0.5 * np.sin(2 * np.pi * 440 * t)

    @patch('librosa.load')
    def test_load_audio(self, mock_load):
        mock_load.return_value = (self.test_tone, self.sr)

        y, sr = self.processor.load_audio("dummy_path.wav")

        mock_load.assert_called_once_with('dummy_path.wav', sr=self.sr)
        self.assertEqual(sr, self.sr)
        np.testing.assert_array_equal(y, self.test_tone)

    def test_split_audio(self):
        y = self.test_tone
        segments = self.processor.split_audio(y)
        expected_segments = int(np.ceil(len(y) / (self.sr * self.duration)))

        self.assertEqual(len(segments), expected_segments)

        for segment in segments[:-1]:
            self.assertEqual(len(segment), int(self.sr * self.duration))

        self.assertTrue(len(segments[-1]) <= int(self.sr * self.duration))


if __name__ == '__main__':
    unittest.main()