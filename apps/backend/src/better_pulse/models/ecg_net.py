import torch
import torch.nn as nn

class ECGNet(nn.Module):
    """
    1D Convolutional Neural Network for processing single-channel ECG signals.
    Input shape:  [Batch, 1, 1000]
    Output shape: [Batch, 2]
    """
    def __init__(self, input_channels: int = 1, num_classes: int = 2):
        super(ECGNet, self).__init__()
        # Convolutional Block 1
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)  # seq length: 250
        
        # Convolutional Block 2
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7, stride=1, padding=3)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)  # seq length: 125
        
        # Convolutional Block 3
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(2)  # seq length: 62
        
        # Global pooling and fully connected layers
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Reduces to [Batch, 64, 1]
        self.fc1 = nn.Linear(64, 32)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        
        x = self.global_pool(x)
        x = x.squeeze(-1)
        
        x = self.relu_fc(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
