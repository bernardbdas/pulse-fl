from pathlib import Path
import torch
from safetensors.torch import save_file
from better_pulse.config import settings
from better_pulse.models.ecg_net import ECGNet

class ModelFactory:
    """
    Factory Pattern for Neural Network Models.
    Coordinates wearable model instantiations and weight initialization.
    """
    @staticmethod
    def create_model(model_name: str = "ECGNet") -> torch.nn.Module:
        if model_name == "ECGNet":
            return ECGNet(input_channels=settings.ECG_INPUT_CHANNELS, num_classes=settings.NUM_CLASSES)
        else:
            raise ValueError(f"Unknown model architecture requested: {model_name}")

    @classmethod
    def initialize_global_model(cls, round_number: int = 0) -> Path:
        """Initializes a model structure and exports it to a safetensors binary file."""
        model = cls.create_model("ECGNet")
        model_path = settings.GLOBAL_MODELS_DIR / f"global_model_round_{round_number}.safetensors"
        save_file(model.state_dict(), str(model_path))
        return model_path
