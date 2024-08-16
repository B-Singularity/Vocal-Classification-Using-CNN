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
    """

    def __init__(self, model, validation_loader, device='cuda'):
        """
        Initializes the FilterPruner with a given model and validation data loader.

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
            A negative value will penalize high latency, influencing pruning decisions.
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

        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                print(f"Evaluating layer {name}")

                num_filters = layer.out_channels
                best_proposal = None
                best_tradeoff = -float('inf')  # Initialize with negative infinity

                for filters_to_prune in range(1, num_filters):
                    pruned_layer = self.prune_filter(name, filters_to_prune)

                    new_metric = self.resource_measurer.measure(self.model, input_tensor)
                    new_accuracy = self.evaluator.evaluation_accuracy(self.model)

                    latency_change = (initial_metric - new_metric) / initial_metric
                    accuracy_change = (self.baseline_accuracy - new_accuracy) / self.baseline_accuracy

                    tradeoff = (accuracy_change / latency_change) if latency_change != 0 else -float('inf')

                    if tradeoff > best_tradeoff and latency_change > target_metric_reduction:
                        best_tradeoff = tradeoff
                        best_proposal = (name, pruned_layer, new_metric, new_accuracy)

                if best_proposal:
                    name, best_layer, new_metric, new_accuracy = best_proposal
                    self.replace_layer(self.model, name, best_layer)
                    initial_metric = new_metric
                    self.baseline_accuracy = new_accuracy

    def prune_filter(self, layer_name, num_filters_to_prune):
        """
        Prunes filters from the specified convolutional layer and replaces it with a new layer.

        Parameters
        ----------
        layer_name : str
            The name of the convolutional layer in the model that will be pruned.
        num_filters_to_prune : int
            The number of filters to remove from the specified layer.

        Returns
        -------
        torch.nn.Conv2d
            The new convolutional layer with the remaining filters.
        """
        layer = dict(self.model.named_modules())[layer_name]
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

        return new_layer

    def replace_layer(self, model, name, new_layer):
        """
        Replaces the specified layer in the model with a new layer.

        Parameters
        ----------
        model : torch.nn.Module
            The model containing the layer to be replaced.
        name : str
            The name of the layer to be replaced.
        new_layer : torch.nn.Module
            The new layer to replace the old layer.

        Returns
        -------
        None
        """
        components = name.split('.')
        for comp in components[:-1]:
            model = getattr(model, comp)
        setattr(model, components[-1], new_layer)
