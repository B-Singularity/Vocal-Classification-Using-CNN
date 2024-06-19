import librosa


class AudioProcessor:
    """
    A class used to process audio data, including loading and splitting audio files.

    Attributes
    ----------
    sr : int
        The sample rate to use when loading audio files.
    duration : float
        The duration (in seconds) to split the audio files into segments.

    Methods
    -------
    load_audio(audio_path):
        Loads an audio file using the specified sample rate.

    split_audio(y):
        Splits the loaded audio into segments of the specified duration.
    """

    def __init__(self, sr, duration):
        """
        Constructs all the necessary attributes for the AudioProcessor object.

        Parameters
        ----------
        sr : int
            The sample rate to use when loading audio files.
        duration : float
            The duration (in seconds) to split the audio files into segments.
        """
        self.sr = sr
        self.duration = duration

    def load_audio(self, audio_path):
        """
        Loads an audio file using the specified sample rate.

        Parameters
        ----------
        audio_path : str
            The path to the audio file to be loaded.

        Returns
        -------
        tuple
            A tuple (y, sr) where y is the audio time series and sr is the sampling rate.
        """
        return librosa.load(audio_path, sr=self.sr)

    def split_audio(self, y):
        """
        Splits the loaded audio into segments of the specified duration.

        Parameters
        ----------
        y : np.ndarray
            The audio time series data.

        Returns
        -------
        list
            A list of audio segments where each segment is a numpy array.
        """
        n_samples = int(self.sr * self.duration)
        return [y[i:i + n_samples] for i in range(0, len(y), n_samples)]


