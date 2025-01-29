import torch
import torch.nn as nn
import torch.nn.functional as F

DEBUG = False

def calculate_padding(kernel_size):
    return ((kernel_size-1)  // 2)    


class SpatialTemporalSEBlock(nn.Module):
    def __init__(self, in_channels, num_electrodes=8, num_timepoints=100, reduction_ratio=16):
        """
        Squeeze-and-Excitation block with separate spatial and temporal attention pathways.

        Args:
            in_channels (int): Number of input channels.
            num_electrodes (int): Number of electrodes (spatial dimension).
            num_timepoints (int): Number of time samples (temporal dimension).
            reduction_ratio (int): Reduction ratio for bottleneck in attention mechanism.
        """
        super(SpatialTemporalSEBlock, self).__init__()
        
        # Channel attention
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc1 = nn.Linear(in_channels, in_channels // reduction_ratio)
        self.channel_fc2 = nn.Linear(in_channels // reduction_ratio, in_channels)
        
        # # Spatial attention pathway
        # self.spatial_pool = nn.AdaptiveAvgPool2d((num_electrodes, 1))  # Pool over time
        # self.spatial_fc1 = nn.Linear(num_electrodes, num_electrodes // reduction_ratio)
        # self.spatial_fc2 = nn.Linear(num_electrodes // reduction_ratio, num_electrodes)
        
        # # Temporal attention pathway
        # self.temporal_pool = nn.AdaptiveAvgPool2d((1, num_timepoints))  # Pool over electrodes
        # self.temporal_fc1 = nn.Linear(num_timepoints, num_timepoints // reduction_ratio)
        # self.temporal_fc2 = nn.Linear(num_timepoints // reduction_ratio, num_timepoints)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the SE block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, num_electrodes, num_timepoints).
        
        Returns:
            torch.Tensor: Output tensor with spatial and temporal attention applied.
        """
        batch_size, in_channels, num_electrodes, num_timepoints = x.size()
        
        # Channel attention
        channel_attention = self.global_pool(x).view(batch_size, in_channels)
        channel_attention = self.relu(self.channel_fc1(channel_attention))
        channel_attention = self.sigmoid(self.channel_fc2(channel_attention)).view(batch_size, in_channels, 1, 1)
        
        # Spatial attention
        # spatial_attention = self.spatial_pool(x).mean(dim=3)  # Pool across time dimension
        # spatial_attention = self.relu(self.spatial_fc1(spatial_attention.view(batch_size, in_channels, num_electrodes)))
        # spatial_attention = self.sigmoid(self.spatial_fc2(spatial_attention)).view(batch_size, in_channels, num_electrodes, 1)
        
        # Temporal attention
        # temporal_attention = self.temporal_pool(x).mean(dim=2)  # Pool across electrode dimension
        # temporal_attention = self.relu(self.temporal_fc1(temporal_attention.view(batch_size, in_channels, num_timepoints)))
        # temporal_attention = self.sigmoid(self.temporal_fc2(temporal_attention)).view(batch_size, in_channels, 1, num_timepoints)
        
        # Residual attention
        # x = x * channel_attention + x * temporal_attention
        print("After SE block:", (x * channel_attention).shape) if DEBUG else None
        return x * channel_attention
    
    
class EEGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temp_kernel, num_electrodes=8, D=2, dropout_rate=0.3):
        super(EEGBlock, self).__init__()
        
        # Temporal Conv Block
        self.batch_norm1 = nn.BatchNorm2d(in_channels)
        self.temp_conv = nn.Conv2d(in_channels, out_channels, temp_kernel, bias=False, padding=(0, calculate_padding(temp_kernel[1])))
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        # Spatial Conv Block
        self.spatial_conv = nn.Conv2d(out_channels, out_channels*D, (num_electrodes, 1), 
                                      groups=out_channels, bias=False)
        
        self.batch_norm3 = nn.BatchNorm2d(out_channels*D)
        
        self.elu = nn.ELU()
        self.drop_out = nn.Dropout2d(dropout_rate)
        
        # Pointwise Convolution
        self.separable_conv_point = nn.Conv2d(out_channels*D, out_channels, 1, bias=False)
    
    def forward(self, x):
        print("Input shape:", x.shape) if DEBUG else None
        x = self.elu(self.batch_norm1(x))
        print("After BatchNorm1:", x.shape) if DEBUG else None
        x = self.temp_conv(x)
        print("After TempConv:", x.shape) if DEBUG else None
        x = self.elu(self.batch_norm2(x))
        print("After BatchNorm2:", x.shape) if DEBUG else None
        x = self.spatial_conv(x)
        print("After SpatialConv:", x.shape) if DEBUG else None
        x = self.elu(self.batch_norm3(x))
        
        x = self.drop_out(x)
        x = self.separable_conv_point(x)
        print("After SeparableConv:", x.shape) if DEBUG else None
        return x
    

class WideResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temp_kernel, num_electrodes):
        super(WideResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.elu = nn.ELU()
        
        self.eegblock = EEGBlock(in_channels, out_channels, temp_kernel, num_electrodes)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        print() if DEBUG else None
        residual = self.bn_shortcut(self.shortcut(x))
        out = self.elu(self.bn1(x))
        out = self.eegblock(out)
        return out + residual
    


class EEGWRN(nn.Module):
    def __init__(self, depth=16, width=4, num_electrodes=8, num_timepoints=100):
        super(EEGWRN, self).__init__()
        self.depth = depth
        self.width = width
        
        n_chans = [16, 16*width, 32*width, 64*width]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        
        self.conv1 = nn.Conv2d(1, n_chans[0], kernel_size=(1, num_timepoints-1), stride=1, padding=(0, calculate_padding(num_timepoints)), bias=False)
        
        self.block1 = self.make_layer(n_chans[0], n_chans[1], n, (1, 64-1), num_electrodes, num_timepoints)
        self.block2 = self.make_layer(n_chans[1], n_chans[2], n, (1, 32-1), num_electrodes, num_timepoints)
        self.block3 = self.make_layer(n_chans[2], n_chans[3], n, (1, 16-1), num_electrodes, num_timepoints)
        
        self.avgpool = nn.AvgPool2d((1,2))
        
        self.bn1 = nn.BatchNorm2d(n_chans[3])
        self.elu = nn.ELU()
        
        self.flatten = nn.Flatten()
        
        self.dense = nn.Sequential(
            nn.Linear(64*width, 256),
            nn.ReLU(),
            nn.Linear(256, 1)    
        )
        
        self.n_channels = n_chans[3]
        self._initialize_weights()
        
    def make_layer(self, in_channels, out_channels, n_layers, temp_kernel, num_electrodes, num_timepoints):
        layers = []
        for _ in range(n_layers):
            layers.append(WideResBlock(in_channels, out_channels, temp_kernel, num_electrodes))
            layers.append(SpatialTemporalSEBlock(out_channels, num_electrodes, num_timepoints))
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        print("Input shape:", x.shape) if DEBUG else None
        out = self.conv1(x)
        print("After Initial Convolutional shape:", out.shape) if DEBUG else None
        out = self.block1(out)
        out = self.avgpool(out)
        out = self.block2(out)
        out = self.avgpool(out)
        out = self.block3(out)
        out = self.elu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        embeddings = self.flatten(out)
        print(out.shape) if DEBUG else None
        print(embeddings.shape) if DEBUG else None
        return embeddings, self.dense(embeddings)