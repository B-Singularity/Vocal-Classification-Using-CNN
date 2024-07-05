import sys
import os
import unittest
import wave
import numpy as np

# 현재 파일의 경로에서 src 디렉토리까지의 절대 경로를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# ConvertToWav 클래스를 import
from preprocessing.convert_to_wav import ConvertToWav


# 간단한 WAV 파일 생성 함수
def create_test_wav(filename):
    sample_rate = 44100  # 44.1 kHz
    duration = 1  # 1 second
    frequency = 440.0  # 440 Hz (A4)

    with wave.open(filename, 'w') as wav_file:
        n_channels = 1
        sampwidth = 2
        n_frames = sample_rate * duration
        comptype = "NONE"
        compname = "not compressed"
        wav_file.setparams((n_channels, sampwidth, sample_rate, n_frames, comptype, compname))

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        data = (np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

        wav_file.writeframes(data.tobytes())


class TestConvertToWav(unittest.TestCase):

    def test_convert_to_wav(self):
        test_dir = "test_directory"
        temp_file_path = os.path.join(test_dir, "test_audio.mp3")

        # Create a directory for the test if it does not exist
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        # Create a temporary test audio file
        create_test_wav(temp_file_path)

        # Create an instance of ConvertToWav
        converter = ConvertToWav(file_path=temp_file_path, folder_path=test_dir)

        # Call the convert_and_save method
        converter.convert_and_save()

        # Assert that the WAV file has been created
        wav_file_path = os.path.splitext(temp_file_path)[0] + ".wav"
        self.assertTrue(os.path.exists(wav_file_path))

        # Clean up: Delete the temporary test files
        os.remove(temp_file_path)
        os.remove(wav_file_path)
        os.rmdir(test_dir)


if __name__ == "__main__":
    unittest.main()
