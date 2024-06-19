import os

class FileSystemHelper:
    """
    A helper class for filesystem operations related to WAV files.

    Methods
    -------
    make_dir(path):
        Creates a directory if it doesn't already exist.

    get_wav_files(folder_path):
        Retrieves a list of paths to WAV files within a specified folder.

    """

    @staticmethod
    def make_dir(path):
        """
        Creates a directory if it doesn't already exist.

        Parameters
        ----------
        path : str
            The path of the directory to be created.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def get_wav_files(folder_path):
        """
        Retrieves a list of paths to WAV files within a specified folder.

        Parameters
        ----------
        folder_path : str
            The path of the folder where WAV files are located.

        Returns
        -------
        list
            A list of paths to WAV files (absolute paths).
        """
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('wav')]
