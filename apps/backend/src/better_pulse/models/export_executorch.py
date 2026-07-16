from pathlib import Path
import torch
from better_pulse.config import settings
from better_pulse.models.factory import ModelFactory

def export_to_executorch(output_path: Path):
    """
    Traces the ECGNet model using PyTorch's torch.export and exports it to ExecuTorch format.
    """
    print(f"Tracing ECGNet with input shape: torch.Size([1, {settings.ECG_INPUT_CHANNELS}, {settings.ECG_SEQUENCE_LENGTH}])...")
    
    model = ModelFactory.create_model("ECGNet")
    model.eval()
    
    example_input = torch.randn(1, settings.ECG_INPUT_CHANNELS, settings.ECG_SEQUENCE_LENGTH)
    
    exported_program = torch.export.export(model, (example_input,))
    print("Model successfully exported to PyTorch ExportedProgram.")
    
    try:
        from executorch.exir import to_edge
        edge_program = to_edge(exported_program)
        et_program = edge_program.to_executorch()
        with open(output_path, "wb") as f:
            f.write(et_program.buffer)
        print(f"ExecuTorch binary successfully saved to: {output_path}")
    except ImportError:
        print("[WARNING]: 'executorch' SDK is not installed in the local Python environment.")
        print("ExportedProgram was validated. Saving PyTorch trace (.pt) as fallback PTE representation.")
        torch.save(exported_program, output_path)
        print(f"Fallback traced model saved to: {output_path}")
