import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.convert_to_mel_spectrogram import ConvertToMelSpectrogram

def main():
    folder_path = '/Users/seong-gyeongjun/Downloads/vocal artist/sia/wav'
    save_path = '/Users/seong-gyeongjun/Downloads/vocal artist/sia/mel'

    sr = 22050
    n_fft = 2048
    duration = 3.0
    n_mels = 128
    hop_length = n_fft // 4

    converter = ConvertToMelSpectrogram(folder_path, save_path, sr, n_fft, duration, n_mels, hop_length)

    converter.convert_folder_to_mel_spectrogram()

if __name__ == '__main__':
    main()
