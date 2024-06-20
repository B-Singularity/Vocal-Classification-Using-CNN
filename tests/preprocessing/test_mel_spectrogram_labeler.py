import unittest
import tempfile
import pandas as pd
from pathlib import Path
from preprocessing.mel_spectrogram_labeler import MelSpectrogramLabeler

class TestMelSpectrogramLabeler(unittest.TestCase):

    def setUp(self):
        # 임시 디렉터리 생성
        self.temp_dir = tempfile.TemporaryDirectory()

        # 가상의 기본 디렉터리 생성
        self.base_dir = Path(self.temp_dir.name) / 'mock_base'
        self.base_dir.mkdir()

        # 가상의 아티스트 디렉터리와 mel 파일 생성
        artist1_dir = self.base_dir / 'artist1'
        artist1_dir.mkdir()
        mel1_dir = artist1_dir / 'mel'
        mel1_dir.mkdir()
        (mel1_dir / 'spectrogram1.png').touch()

        artist2_dir = self.base_dir / 'artist2'
        artist2_dir.mkdir()
        mel2_dir = artist2_dir / 'mel'
        mel2_dir.mkdir()
        (mel2_dir / 'spectrogram2.png').touch()

        # 테스트할 MelSpectrogramLabeler 객체 생성
        self.labeler = MelSpectrogramLabeler(self.base_dir)

        # 예상 경로와 레이블 정의
        self.expected_paths = [
            str(self.base_dir / 'artist1' / 'mel' / 'spectrogram1.png'),
            str(self.base_dir / 'artist2' / 'mel' / 'spectrogram2.png')
        ]
        self.expected_labels = ['artist1', 'artist2']

    def tearDown(self):
        # 임시 디렉터리 정리
        self.temp_dir.cleanup()

    def test_label_spectrogram(self):
        # label_spectrogram 메서드 호출
        self.labeler.label_spectrogram('mel', 'png')

        # 데이터 검증
        self.assertEqual(len(self.labeler.data), 2)
        for entry in self.labeler.data:
            self.assertIn(entry['file_path'], self.expected_paths)
            self.assertIn(entry['label'], self.expected_labels)

    def test_save_to_dataframe(self):
        # label_spectrogram 메서드 호출
        self.labeler.label_spectrogram('mel', 'png')

        # save_to_dataframe 메서드 호출 및 데이터프레임 반환 검증
        df = self.labeler.save_to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        expected_columns = ['file_path', 'label']
        self.assertListEqual(list(df.columns), expected_columns)

        # label_spectrogram 메서드에서 생성된 데이터와 일치하는 값을 검증
        for index, row in df.iterrows():
            self.assertIn(row['file_path'], self.expected_paths)
            self.assertIn(row['label'], self.expected_labels)

    def test_save_to_csv(self):
        # label_spectrogram 메서드 호출
        self.labeler.label_spectrogram('mel', 'png')

        # 임시 CSV 파일 경로 설정
        temp_csv_path = self.temp_dir.name + '/output.csv'

        # save_to_csv 메서드 호출
        self.labeler.save_to_csv(temp_csv_path)

        # CSV 파일 존재 여부 확인
        self.assertTrue(Path(temp_csv_path).exists())

        # CSV 파일 내용 검증
        df = pd.read_csv(temp_csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        expected_columns = ['file_path', 'label']
        self.assertListEqual(list(df.columns), expected_columns)

        # label_spectrogram 메서드에서 생성된 데이터와 일치하는 값을 검증
        for index, row in df.iterrows():
            self.assertIn(row['file_path'], self.expected_paths)
            self.assertIn(row['label'], self.expected_labels)

if __name__ == '__main__':
    unittest.main()
