from abc import ABC, abstractmethod
from typing import List, Dict
import torch
from safetensors.torch import load_file
from pulse_fl.schemas.db_models import ClientContribution

class AggregationStrategy(ABC):
    """
    Strategy Pattern interface for Federated Learning aggregation algorithms.
    """
    @abstractmethod
    def aggregate(self, contributions: List[ClientContribution]) -> Dict[str, torch.Tensor]:
        pass

class FedAvgStrategy(AggregationStrategy):
    """
    Standard Federated Averaging (FedAvg) algorithm.
    Performs weighted averages of client state dicts based on local dataset sizes.
    """
    def aggregate(self, contributions: List[ClientContribution]) -> Dict[str, torch.Tensor]:
        if not contributions:
            raise ValueError("No contributions provided for aggregation.")

        total_samples = sum(c.sample_count for c in contributions)
        if total_samples == 0:
            raise ValueError("Total sample count across contributions is zero.")

        first_weights = load_file(contributions[0].update_file_path)
        aggregated_weights = {}

        # Initialize base tensors with zeros
        for key, tensor in first_weights.items():
            aggregated_weights[key] = torch.zeros_like(tensor, dtype=torch.float32)

        # Weighted aggregate loop
        for c in contributions:
            weight_factor = c.sample_count / total_samples
            client_weights = load_file(c.update_file_path)
            for key in aggregated_weights.keys():
                client_tensor = client_weights[key].to(torch.float32)
                aggregated_weights[key] += client_tensor * weight_factor

        return aggregated_weights
