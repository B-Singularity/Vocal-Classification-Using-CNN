import torch.nn as nn
import torch


class FilterPruner:
    """
    A class to prune filters from convolutional layers in a neural network model.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model containing the layers to be pruned.

    Methods
    -------
    prune_filter(layer_idx, num_filters_to_prune):
        Prunes the specified number of filters from a convolutional layer and replaces it with a new, smaller layer.
    """

    def __init__(self, model):
        """
        Initializes the FilterPruner with a given model.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model containing the layers to be pruned.
        """
        self.model = model

    def prune_filter(self, layer_idx, num_filters_to_prune):
        """
        Prunes filters from the specified convolutional layer and replaces it with a new layer.

        Parameters
        ----------
        layer_idx : int
            The index of the convolutional layer in the model that will be pruned.
        num_filters_to_prune : int
            The number of filters to remove from the specified layer.

        Returns
        -------
        torch.nn.Conv2d
            The new convolutional layer with the remaining filters.
        """
        layer = self.model.layers[layer_idx]
        keep_filters = layer.out_channels - num_filters_to_prune
        indices = torch.argsort(layer.weight.abs().sum(dim=[1, 2, 3]))[num_filters_to_prune:]

        new_layer = nn.Conv2d(
            in_channels=layer.in_channels,
            out_channels=keep_filters,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias is not None
        )

        new_layer.weight.data = layer.weight.data[indices].clone()
        if layer.bias is not None:
            new_layer.bias.data = layer.bias.data[indices].clone()

        self.model.features[layer_idx] = new_layer

        return new_layer
