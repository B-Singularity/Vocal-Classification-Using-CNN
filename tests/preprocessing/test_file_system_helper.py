import unittest
import os
from tempfile import TemporaryDirectory
from preprocessing.file_system_helper import FileSystemHelper


class TestFileSystemHelper(unittest.TestCase):

    def test_make_dir(self):
        """Test the make_dir method."""
        with TemporaryDirectory() as tempdir:
            test_path = os.path.join(tempdir, 'test_directory')
            FileSystemHelper.make_dir(test_path)

            # Verify that the directory was created
            self.assertTrue(os.path.exists(test_path))
            self.assertTrue(os.path.isdir(test_path))

    def test_get_wav_files(self):
        """Test the get_wav_files method."""
        with TemporaryDirectory() as tempdir:
            test_folder = os.path.join(tempdir, 'test_folder')
            os.makedirs(test_folder, exist_ok=True)

            # Create test files
            test_files = ['test1.wav', 'test2.wav', 'test3.txt']
            for filename in test_files:
                open(os.path.join(test_folder, filename), 'a').close()

            # Call the method under test
            wav_files = FileSystemHelper.get_wav_files(test_folder)

            # Verify the result
            expected_files = [os.path.join(test_folder, file) for file in ['test1.wav', 'test2.wav']]
            self.assertEqual(sorted(wav_files), sorted(expected_files))


if __name__ == '__main__':
    unittest.main()
