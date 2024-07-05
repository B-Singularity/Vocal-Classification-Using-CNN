from data.data_splitter import DataSplitter
from data.sound_dataset import SoundDataset
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO
import torch

class TestDataSplitter(unittest.TestCase):

    def setUp(self):
        csv_data = """index,path,label
                      1,path/to/audio1.wav,label1
                      2,path/to/audio2.wav,label1
                      3,path/to/audio3.wav,label1
                      4,path/to/audio4.wav,label2
                      5,path/to/audio5.wav,label2
                      6,path/to/audio6.wav,label2
                      7,path/to/audio7.wav,label3
                      8,path/to/audio8.wav,label3
                      9,path/to/audio9.wav,label3
                      10,path/to/audio10.wav,label3"""
        self.annotations_file = StringIO(csv_data)
        self.dataset = SoundDataset(self.annotations_file)

    @patch('torchaudio.load')
    def test_split_dataset(self, mock_load):
        # Mock torchaudio.load 반환 값 설정
        mock_load.return_value = (torch.zeros(1, 16000), 16000)

        splitter = DataSplitter(self.dataset)
        splitter.split_dataset(train_ratio=0.8)

        self.assertEqual(len(splitter.train_set), 8)
        self.assertEqual(len(splitter.val_set), 2)

    @patch('torchaudio.load')
    def test_get_loaders(self, mock_load):
        # Mock torchaudio.load 반환 값 설정
        mock_load.return_value = (torch.zeros(1, 16000), 16000)

        splitter = DataSplitter(self.dataset)
        splitter.split_dataset(train_ratio=0.8)
        train_loader, val_loader = splitter.get_loaders(train_batch_size=2, val_batch_size=1, num_workers=0)

        train_batches = list(train_loader)
        val_batches = list(val_loader)

        self.assertEqual(len(train_batches), 4)
        self.assertEqual(len(val_batches), 2)


if __name__ == '__main__':
    unittest.main()