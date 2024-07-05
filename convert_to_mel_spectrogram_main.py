import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.convert_to_mel_spectrogram import ConvertToMelSpectrogram

def main():
    folder_path = '/Users/seong-gyeongjun/Downloads/vocal artist/stone/wav'
    save_path = '/Users/seong-gyeongjun/Downloads/vocal artist/stone/mel'

    sr = 22050
    duration = 3.0
    n_mels = 128
    hop_length = int(sr * duration / 4)

    converter = ConvertToMelSpectrogram(folder_path, save_path, sr, duration, n_mels, hop_length)

    converter.convert_folder_to_mel_spectrogram()

if __name__ == '__main__':
    main()
