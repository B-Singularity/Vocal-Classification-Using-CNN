import sys, os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch.nn as nn
import torch
from resource_measurement import ResourceMeasurement
from evaluations.metrics import Evaluation


class FilterPruner:
    """
    A class to prune filters from convolutional layers in a neural network model.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model containing the layers to be pruned.
    device : str
        The device ('cuda' or 'cpu') to run the pruning process on.
    resource_measurer : ResourceMeasurement
        A utility to measure the resource usage (e.g., latency) of the model.
    evaluator : Evaluation
        An instance of the Evaluation class to evaluate model accuracy.
    baseline_accuracy : float
        The baseline accuracy of the model before pruning.

    Methods
    -------
    prune_layerwise(input_tensor, target_metric_reduction=0.05, w=-0.15, target_latency=None, reward_threshold=1.0):
        Prunes filters from the model's convolutional layers layer by layer while evaluating performance.

    prune_filter(layer_idx, num_filters_to_prune):
        Prunes the specified number of filters from a convolutional layer and replaces it with a new, smaller layer.

    _update_next_layer(next_layer, keep_filters):
        Adjusts the next convolutional layer to match the pruned output channels.
    """

    def __init__(self, model, validation_loader, device='cuda'):
        """
        Initializes the FilterPruner with a given model.

        Parameters
        ----------
        model : torch.nn.Module
            The neural network model containing the layers to be pruned.
        validation_loader : torch.utils.data.DataLoader
            The DataLoader for the validation dataset used to evaluate model accuracy.
        device : str, optional
            The device to run the pruning process on (default is 'cuda').
        """
        self.model = model
        self.device = device
        self.resource_measurer = ResourceMeasurement(metric='latency')
        self.evaluator = Evaluation(validation_loader, device)
        self.baseline_accuracy = None

    def prune_layerwise(self,
                        input_tensor,
                        target_metric_reduction=0.05,
                        w=-0.15,
                        target_latency=None,
                        reward_threshold=1.0):
        """
        Prunes filters from the model's convolutional layers one layer at a time, evaluating performance after each step.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor used to measure the model's performance (e.g., latency).
        target_metric_reduction : float, optional
            The target reduction in the performance metric (default is 0.05 for 5%).
        w : float, optional
            The weight factor used in the reward calculation (default is -0.15).
        target_latency : float, optional
            The target latency to achieve. If None, it's set to the initial latency (default is None).
        reward_threshold : float, optional
            The threshold for the reward value to decide when to stop pruning (default is 1.0).

        Returns
        -------
        None
        """
        initial_metric = self.resource_measurer.measure(self.model, input_tensor)
        self.baseline_accuracy = self.evaluator.evaluation_accuracy(self.model)

        if target_latency is None:
            target_latency = initial_metric

        for layer_idx, layer in enumerate(self.model.layers):
            if isinstance(layer, nn.Conv2d):
                print(f"Pruning layer {layer_idx}")

                num_filters = layer.out_channels

                for filters_to_prune in range(1, num_filters):
                    pruned_layer = self.prune_filter(layer_idx, filters_to_prune)

                    new_metric = self.resource_measurer.measure(self.model, input_tensor)
                    new_accuracy = self.evaluator.evaluation_accuracy(self.model)

                    latency_change = (initial_metric - new_metric) / initial_metric
                    accuracy_change = (self.baseline_accuracy - new_accuracy) / self.baseline_accuracy

                    reward = (new_accuracy / self.baseline_accuracy) * ((new_metric / target_latency) ** w)

                    if latency_change > target_metric_reduction or reward < reward_threshold:
                        self.model.layers[layer_idx] = layer
                        break
                    else:
                        initial_metric = new_metric
                        self.baseline_accuracy = new_accuracy

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
        filter_norms = torch.norm(layer.weight.data, p=2, dim=[1, 2, 3])
        indices = torch.argsort(filter_norms)[num_filters_to_prune:]

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

        self.model.layers[layer_idx] = new_layer

        if layer_idx + 1 < len(self.model.layers):
            next_layer = self.model.layers[layer_idx + 1]
            if isinstance(next_layer, nn.Conv2d):
                self._update_next_layer(next_layer, keep_filters)

        return new_layer

    def _update_next_layer(self, next_layer, keep_filters):
        """
        Adjusts the next convolutional layer to match the pruned output channels.

        Parameters
        ----------
        next_layer : torch.nn.Conv2d
            The convolutional layer immediately following the pruned layer.
        keep_filters : int
            The number of output filters kept from the pruned layer.
        """
        new_next_layer = nn.Conv2d(
            in_channels=keep_filters,
            out_channels=next_layer.out_channels,
            kernel_size=next_layer.kernel_size,
            stride=next_layer.stride,
            padding=next_layer.padding,
            bias=next_layer.bias is not None
        )

        # Update the weights of the next layer to match the pruned input channels
        new_next_layer.weight.data = next_layer.weight.data[:, :keep_filters].clone()
        if next_layer.bias is not None:
            new_next_layer.bias.data = next_layer.bias.data.clone()

        self.model.layers[self.model.layers.index(next_layer)] = new_next_layer
