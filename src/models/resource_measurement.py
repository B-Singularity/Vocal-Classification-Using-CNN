import torch
import time


class ResourceMeasurement:
    """
    A class to measure the resource usage of a model based on a specified metric.

    Attributes:
    ----------
    metric : str
        The metric to measure ('latency' or 'memory'). Default is 'latency'.
    metric_functions : dict
        A dictionary that maps metrics to their corresponding measurement functions.

    Methods:
    -------
    _run_model(model, input_tensor):
        Runs the model on the input tensor in evaluation mode.

    _measure_latency(model, input_tensor):
        Measures the latency (time taken) for the model to process the input tensor.

    _measure_memory_footprint(model, input_tensor):
        Measures the peak GPU memory usage when the model processes the input tensor.

    measure(model, input_tensor):
        Measures the resource usage based on the specified metric.
    """

    def __init__(self, metric='latency'):
        """
        Initializes the ResourceMeasurement class with the specified metric.

        Parameters:
        ----------
        metric : str, optional
            The metric to measure. Can be 'latency' or 'memory'. Default is 'latency'.
        """
        self.metric = metric
        self.metric_functions = {
            'latency': self._measure_latency,
            'memory': self._measure_memory_footprint
        }

    def _run_model(self, model, input_tensor):
        """
        Runs the model in evaluation mode with the provided input tensor.

        Parameters:
        ----------
        model : torch.nn.Module
            The neural network model to evaluate.
        input_tensor : torch.Tensor
            The input tensor to feed into the model.
        """
        model.eval()  # Switch to evaluation mode (disabling dropout, batchnorm, etc.)
        with torch.no_grad():
            _ = model(input_tensor)

    def _measure_latency(self, model, input_tensor):
        """
        Measures the latency of the model for processing the input tensor.

        Parameters:
        ----------
        model : torch.nn.Module
            The neural network model to evaluate.
        input_tensor : torch.Tensor
            The input tensor to feed into the model.

        Returns:
        -------
        float
            The time taken (in seconds) for the model to process the input tensor.
        """
        start_time = time.time()
        self._run_model(model, input_tensor)
        end_time = time.time()

        latency = end_time - start_time
        return latency

    def _measure_memory_footprint(self, model, input_tensor):
        """
        Measures the peak memory usage on the GPU while the model processes the input tensor.

        Parameters:
        ----------
        model : torch.nn.Module
            The neural network model to evaluate.
        input_tensor : torch.Tensor
            The input tensor to feed into the model.

        Returns:
        -------
        int
            The peak memory usage in bytes on the GPU.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        self._run_model(model, input_tensor)

        memory_used = torch.cuda.max_memory_allocated()
        return memory_used

    def measure(self, model, input_tensor):
        """
        Measures the resource usage based on the specified metric.

        Parameters:
        ----------
        model : torch.nn.Module
            The neural network model to evaluate.
        input_tensor : torch.Tensor
            The input tensor to feed into the model.

        Returns:
        -------
        float or int
            The measured resource usage, depending on the metric.

        Raises:
        -------
        ValueError
            If an unsupported metric is specified.
        """
        if self.metric in self.metric_functions:
            return self.metric_functions[self.metric](model, input_tensor)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
