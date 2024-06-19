import unittest
import os
import tempfile
import numpy as np
import soundfile as sf
import librosa
from unittest.mock import patch
from preprocessing.convert_to_mel_spectrogram import ConvertToMelSpectrogram
from preprocessing.audio_processor import AudioProcessor
from preprocessing.mel_spectrogram_generator import MelSpectrogramGenerator

class TestConvertToMelSpectrogram(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()  # 임시 디렉토리 생성
        self.audio_files = [
            self.create_temp_audio_file('audio1.wav', duration=5, sr=22050),
            self.create_temp_audio_file('audio2.wav', duration=3, sr=22050),
            self.create_temp_audio_file('audio3.wav', duration=7, sr=22050)
        ]
        self.converter = ConvertToMelSpectrogram(
            self.temp_dir,  # 임시 디렉토리 사용
            self.temp_dir,  # 저장 디렉토리도 임시 디렉토리로 설정
            sr=22050,
            duration=5.0,
            n_mels=128,
            hop_length=512
        )

    def tearDown(self):
        # 테스트 종료 후 임시 파일 삭제
        for audio_file in self.audio_files:
            os.remove(audio_file)

    def create_temp_audio_file(self, filename, duration, sr):
        # 임시 WAV 파일 생성
        audio_path = os.path.join(self.temp_dir, filename)
        y = np.random.randn(int(duration * sr))
        sf.write(audio_path, y, sr)  # soundfile로 WAV 파일 생성
        return audio_path

    @patch('librosa.load')
    @patch.object(MelSpectrogramGenerator, 'generate_and_save')
    def test_convert_folder_to_mel_spectrogram(self, mock_generate_and_save, mock_librosa_load):
        # 모의 처리된 librosa.load 함수 설정
        mock_librosa_load.side_effect = self.mock_librosa_load

        # convert_folder_to_mel_spectrogram 메서드 호출
        self.converter.convert_folder_to_mel_spectrogram()

        # generate_and_save가 적절한 파일명으로 호출되었는지 확인
        expected_calls = [
            unittest.mock.call(np.array([0.0]), 'audio1_part0.png'),
            unittest.mock.call(np.array([0.0]), 'audio2_part0.png'),
            unittest.mock.call(np.array([0.0]), 'audio3_part0.png'),
        ]
        mock_generate_and_save.assert_has_calls(expected_calls, any_order=True)

    def mock_librosa_load(self, audio_path, sr=None):
        # 임의의 오디오 데이터 반환
        return np.array([0.0]), 22050

if __name__ == '__main__':
    unittest.main()


