import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameCNN(nn.Module):
    def __init__(self, in_channels=1024, bottleneck_channels=256, out_dim=256):
        """
        A CNN to process one frame. It first reduces the channel dimension using a 1x1 convolution,
        then applies a spatial convolution, global pooling, and a final FC to produce a fixed-length vector.
        """
        super(FrameCNN, self).__init__()
        # 1x1 convolution to reduce channels from 1024 -> bottleneck_channels (e.g. 256)
        self.conv_reduce = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1)
        self.bn_reduce = nn.BatchNorm2d(bottleneck_channels)
        
        # A spatial convolution operating in the reduced channel space
        self.conv_spatial = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)
        self.bn_spatial = nn.BatchNorm2d(bottleneck_channels)
        
        # Global average pooling to collapse spatial dimensions
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # A fully connected layer mapping from bottleneck_channels to the desired output dimension
        self.fc = nn.Linear(bottleneck_channels, out_dim)
    
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 1024, 14, 14)
        """
        # Reduce channel dimension using 1x1 conv
        x = F.relu(self.bn_reduce(self.conv_reduce(x)))   # shape: (batch, bottleneck_channels, 14, 14)
        # Apply spatial convolution
        x = F.relu(self.bn_spatial(self.conv_spatial(x)))    # shape remains: (batch, bottleneck_channels, 14, 14)
        # Global average pooling over spatial dimensions
        x = self.avgpool(x)                                  # shape: (batch, bottleneck_channels, 1, 1)
        x = x.view(x.size(0), -1)                            # flatten to (batch, bottleneck_channels)
        # Map to final per-frame feature vector
        x = self.fc(x)                                     # shape: (batch, out_dim)
        return x

class EmbeddingVideoCNN(nn.Module):
    def __init__(self, frame_feature_dim=256, num_frames=8, action_dim=7):
        """
        This module processes a video (a sequence of frames) by applying FrameCNN to each frame and then
        combining the results via an MLP.
        """
        super(EmbeddingVideoCNN, self).__init__()
        self.num_frames = num_frames
        # Process each frame with the efficient FrameCNN
        self.frame_cnn = FrameCNN(in_channels=1024, bottleneck_channels=256, out_dim=frame_feature_dim)
        
        # MLP to process the concatenated per-frame features
        self.mlp = nn.Sequential(
            nn.Linear(num_frames * frame_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 8, 1024, 14, 14)
        """
        batch_size = x.size(0)
        # Process all frames at once by merging the batch and frame dimensions
        x = x.view(batch_size * self.num_frames, 1024, 14, 14)
        
        # Get per-frame features
        frame_features = self.frame_cnn(x)  # shape: (batch_size * num_frames, frame_feature_dim)
        
        # Reshape to (batch_size, num_frames * frame_feature_dim)
        frame_features = frame_features.view(batch_size, -1)
        
        # Pass the concatenated features through the MLP to produce the final 7-dimensional action output
        action_output = self.mlp(frame_features)  # shape: (batch_size, action_dim)
        return action_output

# Example usage:
if __name__ == '__main__':
    dummy_video = torch.randn(4, 8, 1024, 14, 14)  # 4 videos, 8 frames each, with the given dimensions
    model = EmbeddingVideoCNN(frame_feature_dim=256, num_frames=8, action_dim=7)
    output = model(dummy_video)
    print("Output shape:", output.shape)  # Expected: (4, 7)
