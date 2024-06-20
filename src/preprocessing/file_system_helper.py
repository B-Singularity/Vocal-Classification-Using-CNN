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
    def ensure_directory_exists(path):
        """
        Ensures that a directory exists at the given path. Creates it if necessary.

        Parameters
        ----------
        path : str
            The path of the directory to ensure existence.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def get_files_by_extension(folder_path, extension):
        """
        Retrieves a list of paths to files with a specific extension within a folder.

        Parameters
        ----------
        folder_path : str
            The path of the folder where files are located.
        extension : str
            The extension of files to retrieve (e.g., 'wav', 'png').

        Returns
        -------
        list
            A list of paths to files with the specified extension (absolute paths).
        """
        return [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(extension)
                and os.path.isfile(os.path.join(folder_path, file))]
