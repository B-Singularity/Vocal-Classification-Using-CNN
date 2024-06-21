import unittest
import pandas as pd
import os
from unittest.mock import patch
from io import BytesIO
import torchaudio
import torch
from data.sound_dataset import SoundDataset  # 클래스가 정의된 파일 이름에 맞게 조정

class TestSoundDataset(unittest.TestCase):
    def setUp(self):
        # 가상의 주석 데이터 CSV 파일 생성
        self.annotations = pd.DataFrame([
            [0, "fake_path_1.wav", "label_1"],
            [1, "fake_path_2.wav", "label_2"]
        ], columns=["id", "path", "label"])
        self.annotations_file = "test_annotations.csv"
        self.annotations.to_csv(self.annotations_file, index=False)

        # SoundDataset 객체 생성
        self.dataset = SoundDataset(self.annotations_file)

    def tearDown(self):
        # 테스트 후 생성된 파일을 삭제합니다.
        os.remove(self.annotations_file)

    def test_len(self):
        self.assertEqual(len(self.dataset), 2)

    def test_getitem(self):
        with patch('torchaudio.load') as mocked_load:
            mocked_load.return_value = (torch.zeros((1, 16000)), 16000)
            signal, label = self.dataset[0]
            mocked_load.assert_called_once_with("fake_path_1.wav")
            self.assertTrue(torch.equal(signal, torch.zeros((1, 16000))))
            self.assertEqual(label, "label_1")

    def test_get_audio_path(self):
        path = self.dataset._get_audio_path(0)
        self.assertEqual(path, "fake_path_1.wav")

    def test_get_audio_label(self):
        label = self.dataset._get_audio_label(0)
        self.assertEqual(label, "label_1")

if __name__ == '__main__':
    unittest.main()
