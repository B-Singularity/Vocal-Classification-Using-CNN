from torch.utils.data import Dataset, DataLoader, random_split

class DataSplitter:
    """
    A class to split a dataset into training and validation sets.

    Attributes:
    dataset (Dataset): The dataset to be split.
    train_set (Dataset): The training subset.
    val_set (Dataset): The validation subset.

    Methods:
    split_dataset(train_ratio): Splits the dataset into training and validation sets.
    get_loaders(train_batch_size, val_batch_size, num_workers): Returns DataLoaders for the splits.
    """

    def __init__(self, dataset):
        """
        Initializes the DataSplitter with the given dataset.

        Parameters:
        dataset (Dataset): The dataset to be split.
        """
        self.dataset = dataset
        self.train_set = None
        self.val_set = None

    def split_dataset(self, train_ratio):
        """
        Splits the dataset into training and validation sets based on the given ratio.

        Parameters:
        train_ratio (float): The ratio of the training set. Default is 0.8.
        """
        train_size = int(train_ratio * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_set, self.val_set = random_split(self.dataset, [train_size, val_size])

    def get_loaders(self, train_batch_size, val_batch_size, num_workers):
        """
        Returns DataLoaders for the training and validation sets.

        Parameters:
        train_batch_size (int): Batch size for the training set. Default is 32.
        val_batch_size (int): Batch size for the validation set. Default is 32.
        num_workers (int): Number of worker threads to use for data loading. Default is 2.

        Returns:
        tuple: A tuple containing the training and validation DataLoaders.
        """
        train_loader = DataLoader(self.train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(self.val_set, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader
