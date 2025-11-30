import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # Block 1: 64x64 -> 32x32
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        #self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2) # Output: 32x32

        # Block 2: 32x32 -> 16x16
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        #self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2) # Output: 16x16

        # Block 3: 16x16 -> 8x8
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        #self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2) # Output: 8x8

        # --- Flattens to 128 * 8 * 8 = 8192 ---
        self.fc1 = nn.Linear(128 * 8 * 8, 512) # Note the change in input size!
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        #x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x) # No ReLU on final layer for CrossEntropyLoss

        return x