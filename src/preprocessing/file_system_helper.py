import os


class FileSystemHelper:
    @staticmethod
    def make_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def get_wav_files(folder_path):
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('wav')]
    