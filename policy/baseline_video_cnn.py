import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameCNN(nn.Module):
    def __init__(self, in_channels=3, out_dim=256):
        """
        A CNN for processing a single frame of size 3 x 224 x 224.
        The network reduces the spatial dimensions through several layers
        and outputs a fixed-length feature vector.
        """
        super(FrameCNN, self).__init__()
        # First convolution: 7x7 kernel, stride 2 (like early layers in ResNet)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1   = nn.BatchNorm2d(64)
        # A max pooling layer to further reduce spatial dimensions
        self.pool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Second convolution: 3x3 kernel, stride 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(128)
        # Third convolution: 3x3 kernel, stride 2; output spatial dimensions should be near 14x14
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3   = nn.BatchNorm2d(256)
        # Global average pooling to collapse the spatial dimensions to 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer mapping the 256 features to the desired output dimension
        self.fc = nn.Linear(256, out_dim)
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 3, 224, 224)
        """
        x = F.relu(self.bn1(self.conv1(x)))  # -> (batch, 64, 112, 112)
        x = self.pool(x)                     # -> (batch, 64, 56, 56)
        x = F.relu(self.bn2(self.conv2(x)))  # -> (batch, 128, 28, 28)
        x = F.relu(self.bn3(self.conv3(x)))  # -> (batch, 256, 14, 14)
        x = self.avgpool(x)                  # -> (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)            # flatten to (batch, 256)
        x = self.fc(x)                       # -> (batch, out_dim)
        return x

class VideoCNN(nn.Module):
    def __init__(self, frame_feature_dim=256, num_frames=8, action_dim=7):
        """
        This module processes a video (a sequence of frames) by applying FrameCNN
        to each frame individually, concatenating the per-frame features, and then
        processing them via an MLP to produce the final 7-dimensional action output.
        """
        super(VideoCNN, self).__init__()
        self.num_frames = num_frames
        # CNN to process each frame
        self.frame_cnn = FrameCNN(in_channels=3, out_dim=frame_feature_dim)
        # MLP to process concatenated per-frame features
        self.mlp = nn.Sequential(
            nn.Linear(num_frames * frame_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, num_frames, 3, 224, 224)
        """
        batch_size = x.size(0)
        # Merge the batch and frame dimensions so that we process all frames at once:
        # New shape: (batch_size * num_frames, 3, 224, 224)
        x = x.view(batch_size * self.num_frames, 3, 224, 224)
        # Get per-frame features
        frame_features = self.frame_cnn(x)  # -> (batch_size * num_frames, frame_feature_dim)
        # Reshape back to (batch_size, num_frames * frame_feature_dim)
        frame_features = frame_features.view(batch_size, -1)
        # Pass through the MLP to get the final action prediction
        action_output = self.mlp(frame_features)  # -> (batch_size, action_dim)
        return action_output

# Example usage:
if __name__ == '__main__':
    # Create a dummy batch: 4 videos, each with 8 frames of 3 x 224 x 224 images.
    dummy_video = torch.randn(4, 16, 3, 224, 224)
    model = VideoCNN(frame_feature_dim=256, num_frames=16, action_dim=7)
    output = model(dummy_video)
    print("Output shape:", output.shape)  # Expected: (4, 7)
