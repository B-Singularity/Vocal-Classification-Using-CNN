import torch
import torch.nn as nn
import torch.optim as optim

class Finetuner:
    """
    A class to finetune a neural network model for a specified number of steps.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model to be finetuned.
    data_loader : torch.utils.data.DataLoader
        The DataLoader for the dataset used for finetuning.
    device : str
        The device ('cuda' or 'cpu') to run the finetuning process on.
    optimizer : torch.optim.Optimizer
        The optimizer used for finetuning.
    criterion : torch.nn.Module
        The loss function used for finetuning.

    Methods
    -------
    finetune(steps):
        Finetunes the model for the specified number of steps.
    """

    def __init__(self, model, data_loader, device='cuda', lr=0.001, weight_decay=0.0005):
        """
        Initializes the Finetuner with a given model, data loader, and other parameters.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model to be finetuned.
        data_loader : torch.utils.data.DataLoader
            The DataLoader for the dataset used for finetuning.
        device : str, optional
            The device to run the finetuning process on (default is 'cuda').
        lr : float, optional
            The learning rate for the optimizer (default is 0.001).
        weight_decay : float, optional
            The weight decay (regularization) for the optimizer (default is 0.0005).
        """
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()  # Change according to your problem

    def finetune(self, steps):
        """
        Finetunes the model for a given number of steps.

        Parameters
        ----------
        steps : int
            The number of steps to finetune the model.

        Returns
        -------
        None
        """
        self.model.to(self.device)
        self.model.train()

        for _ in range(steps):
            for inputs, targets in self.data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
