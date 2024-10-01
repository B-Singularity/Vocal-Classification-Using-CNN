from pathlib import Path
from preprocessing.mel_spectrogram_labeler import MelSpectrogramLabeler

def main():
    # Define the base directory where your Mel spectrogram data is located
    base_directory = '/Users/seong-gyeongjun/Downloads/vocal artist'

    # Create an instance of MelSpectrogramLabeler
    labeler = MelSpectrogramLabeler(base_directory)

    # Define the directory name containing Mel spectrogram files and their extension
    mel_dir_name = 'mel'
    extension = 'png'

    # Label the spectrograms
    labeler.label_spectrogram(mel_dir_name, extension)

    # Save the labeled data to a CSV file
    output_csv_path = '/Users/seong-gyeongjun/Downloads/vocal_label/output.csv'
    labeler.save_to_csv(output_csv_path)

    # Optionally, you can also save the labeled data to a DataFrame
    labeled_df = labeler.save_to_dataframe()

    # Print or use the labeled DataFrame as needed
    print(labeled_df.head())

if __name__ == "__main__":
    main()