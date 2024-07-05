import os
import torchaudio

class ConvertToWav:
    def __init__(self, file_path, folder_path):
        self.file_path = file_path
        self.folder_path = folder_path

    def convert_and_save(self):
        waveform, sr = torchaudio.load(self.file_path)
        wav_file_path = os.path.splitext(self.file_path)[0] + ".wav"
        torchaudio.save(wav_file_path, waveform, sr)

    def is_supported_file(self, file_name):
        supported_extensions = ['.mp3', '.mp4', '.wav', '.flac', '.ogg']
        return any(file_name.lower().endswith(ext) for ext in supported_extensions)

    def convert_files(self):
        if not os.path.exists(self.folder_path):
            raise ValueError(f"{self.folder_path}은(는) 폴더가 아닙니다.")

        file_list = os.listdir(self.folder_path)

        for file in file_list:
            file_path = os.path.join(self.folder_path, file)
            if os.path.isfile(file_path) and self.is_supported_file(file):
                self.file_path = file_path
                self.convert_and_save()
