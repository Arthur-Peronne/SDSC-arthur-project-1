# src/models/ae_models.py
"""
3D autoencoder architectures for cardiac MRI representation learning.
"""

import torch.nn as nn


class Conv3DBlock(nn.Module):
    """
    3D convolutional block:
    Conv3D -> InstanceNorm3D -> ReLU -> Conv3D -> InstanceNorm3D -> ReLU
    Optionally followed by MaxPool3D for downsampling.
    """

    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = downsample
        if self.downsample:
            self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        if self.downsample:
            x = self.pool(x)

        return x


class UpConv3DBlock(nn.Module):
    """
    3D decoder block:
    ConvTranspose3D (upsampling) -> Conv3D -> InstanceNorm3D -> ReLU -> Conv3D -> InstanceNorm3D -> ReLU
    No skip connections.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv1 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm2 = nn.InstanceNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upconv(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        return x


class AutoEncoder3D_Current(nn.Module):
    """
    Original AE model:
    (1,32,128,128)
    -> (8,16,64,64)
    -> (16,8,32,32)
    -> (32,4,16,16)
    -> bottleneck conv -> (64,4,16,16)
    -> flatten 65536 -> latent_dim
    """

    def __init__(self, latent_dim=10, input_shape=(1, 32, 128, 128)):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder
        self.enc1 = Conv3DBlock(in_channels=1, out_channels=8, downsample=True)
        self.enc2 = Conv3DBlock(in_channels=8, out_channels=16, downsample=True)
        self.enc3 = Conv3DBlock(in_channels=16, out_channels=32, downsample=True)

        self.bottleneck_conv = Conv3DBlock(in_channels=32, out_channels=64, downsample=False)

        self.feature_shape = (64, 4, 16, 16)
        flattened_size = 64 * 4 * 16 * 16  # 65536

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        self.dec1 = UpConv3DBlock(in_channels=64, out_channels=32)
        self.dec2 = UpConv3DBlock(in_channels=32, out_channels=16)
        self.dec3 = UpConv3DBlock(in_channels=16, out_channels=8)

        self.final_conv = nn.Conv3d(
            in_channels=8,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)              # -> (B, 8, 16, 64, 64)
        x = self.enc2(x)              # -> (B, 16, 8, 32, 32)
        x = self.enc3(x)              # -> (B, 32, 4, 16, 16)
        x = self.bottleneck_conv(x)   # -> (B, 64, 4, 16, 16)

        x = self.flatten(x)           # -> (B, 65536)
        x = self.dropout(x)
        z = self.fc_enc(x)            # -> (B, latent_dim)
        return z

    def decode(self, z):
        x = self.fc_dec(z)            # -> (B, 65536)
        x = self.dropout(x)
        x = x.view(-1, *self.feature_shape)

        x = self.dec1(x)              # -> (B, 32, 8, 32, 32)
        x = self.dec2(x)              # -> (B, 16, 16, 64, 64)
        x = self.dec3(x)              # -> (B, 8, 32, 128, 128)

        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class AutoEncoder3D_FCDeep(nn.Module):
    """
    Model A:
    Progressive compression down to (128,1,4,4),
    then flatten -> latent vector -> linear decode.
    """

    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128), dropout_rate=0.0):
        super().__init__()

        self.latent_dim = latent_dim
        self.input_shape = input_shape

        # Encoder
        self.enc1 = Conv3DBlock(1, 8, downsample=True)      # -> (8,16,64,64)
        self.enc2 = Conv3DBlock(8, 16, downsample=True)     # -> (16,8,32,32)
        self.enc3 = Conv3DBlock(16, 32, downsample=True)    # -> (32,4,16,16)
        self.enc4 = Conv3DBlock(32, 64, downsample=True)    # -> (64,2,8,8)

        # Last compression without isotropic pooling because depth=2
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )                                                   # -> (128,2,8,8)

        self.final_down = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2
        )                                                   # -> (128,1,4,4)

        self.feature_shape = (128, 1, 4, 4)
        flattened_size = 128 * 1 * 4 * 4  # 2048

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_rate)   # ← nouveau
        self.fc_enc = nn.Linear(flattened_size, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim, flattened_size)

        self.initial_up = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=128,
            kernel_size=2,
            stride=2
        )                                                   # -> (128,2,8,8)

        self.dec1 = UpConv3DBlock(128, 64)   # -> (64,4,16,16)
        self.dec2 = UpConv3DBlock(64, 32)    # -> (32,8,32,32)
        self.dec3 = UpConv3DBlock(32, 16)    # -> (16,16,64,64)
        self.dec4 = UpConv3DBlock(16, 8)     # -> (8,32,128,128)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.bottleneck_conv(x)
        x = self.final_down(x)
        x = self.flatten(x)
        x = self.dropout(x)
        z = self.fc_enc(x)
        return z

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.dropout(x)  
        x = x.view(-1, *self.feature_shape)
        x = self.initial_up(x)
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

        
class AutoEncoder3D_Conv(nn.Module):
    """
    Model B:
    Fully convolutional bottleneck with shape (C,1,2,2),
    where latent_dim = 4 * C.
    No linear layers.
    """

    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128)):
        super().__init__()

        if latent_dim % 4 != 0:
            raise ValueError("For AutoEncoder3D_Conv, latent_dim must be a multiple of 4.")

        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.latent_channels = latent_dim // 4

        # Encoder
        self.enc1 = Conv3DBlock(1, 8, downsample=True)      # -> (8,16,64,64)
        self.enc2 = Conv3DBlock(8, 16, downsample=True)     # -> (16,8,32,32)
        self.enc3 = Conv3DBlock(16, 32, downsample=True)    # -> (32,4,16,16)
        self.enc4 = Conv3DBlock(32, 64, downsample=True)    # -> (64,2,8,8)

        # Bottleneck reduction:
        # (64,2,8,8) -> (128,2,8,8) -> (128,1,4,4) -> (C,1,2,2)
        self.pre_latent = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )

        self.reduce_to_1x4x4 = nn.Conv3d(
            in_channels=128,
            out_channels=128,
            kernel_size=(2, 2, 2),
            stride=(2, 2, 2)
        )   # (128,2,8,8) -> (128,1,4,4)

        self.reduce_to_latent = nn.Conv3d(
            in_channels=128,
            out_channels=self.latent_channels,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )   # (128,1,4,4) -> (C,1,2,2)

        # Decoder bottleneck inverse:
        # (C,1,2,2) -> (128,1,4,4) -> (128,2,8,8)
        self.expand_from_latent = nn.Sequential(
            nn.ConvTranspose3d(
                self.latent_channels,
                128,
                kernel_size=(1, 2, 2),
                stride=(1, 2, 2)
            ),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )   # -> (128,1,4,4)

        self.expand_to_2x8x8 = nn.Sequential(
            nn.ConvTranspose3d(
                128,
                128,
                kernel_size=(2, 2, 2),
                stride=(2, 2, 2)
            ),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )   # -> (128,2,8,8)

        self.dec1 = UpConv3DBlock(128, 64)   # -> (64,4,16,16)
        self.dec2 = UpConv3DBlock(64, 32)    # -> (32,8,32,32)
        self.dec3 = UpConv3DBlock(32, 16)    # -> (16,16,64,64)
        self.dec4 = UpConv3DBlock(16, 8)     # -> (8,32,128,128)

        self.final_conv = nn.Conv3d(8, 1, kernel_size=3, stride=1, padding=1)
        self.final_activation = nn.Sigmoid()

    def encode_tensor(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.pre_latent(x)           # -> (128,2,8,8)
        x = self.reduce_to_1x4x4(x)      # -> (128,1,4,4)
        z_tensor = self.reduce_to_latent(x)  # -> (B,C,1,2,2)

        return z_tensor

    def encode(self, x):
        z_tensor = self.encode_tensor(x)
        z = z_tensor.flatten(start_dim=1)  # -> (B, latent_dim)
        return z

    def decode_from_tensor(self, z_tensor):
        x = self.expand_from_latent(z_tensor)   # -> (128,1,4,4)
        x = self.expand_to_2x8x8(x)             # -> (128,2,8,8)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)

        x = self.final_conv(x)
        x = self.final_activation(x)
        return x

    def decode(self, z):
        z_tensor = z.view(-1, self.latent_channels, 1, 2, 2)
        x = self.decode_from_tensor(z_tensor)
        return x

    def forward(self, x):
        z_tensor = self.encode_tensor(x)
        z = z_tensor.flatten(start_dim=1)
        x_recon = self.decode_from_tensor(z_tensor)
        return x_recon, z

class AutoEncoder3D_Linear(nn.Module):
    """
    Purely linear autoencoder — no convolutions, no activations.
 
    Encoder: flatten -> nn.Linear(input_size, latent_dim)
    Decoder: nn.Linear(latent_dim, input_size) -> reshape
 
    With MSE loss and no non-linearities, this is theoretically equivalent
    to PCA: the learned subspace should converge to the top-k principal
    components (up to rotation within the subspace).
 
    Input shape : (B, 1, 32, 128, 128)
    Latent shape : (B, latent_dim)
    """
 
    def __init__(self, latent_dim=20, input_shape=(1, 32, 128, 128)):
        super().__init__()
 
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.input_size = 1
        for dim in input_shape:
            self.input_size *= dim  # 1 * 32 * 128 * 128 = 524288
 
        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(self.input_size, latent_dim, bias=True)
        self.fc_dec = nn.Linear(latent_dim, self.input_size, bias=True)
 
    def encode(self, x):
        x = self.flatten(x)       # -> (B, 524288)
        z = self.fc_enc(x)        # -> (B, latent_dim)
        return z
 
    def decode(self, z):
        x = self.fc_dec(z)                     # -> (B, 524288)
        x = x.view(-1, *self.input_shape)      # -> (B, 1, 32, 128, 128)
        return x
 
    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

def build_autoencoder(model_name, latent_dimensions, dropout_rate=0.0):
    """
    Build one of the available AE models.
    """
    if model_name == "AE3dCurrent":
        return AutoEncoder3D_Current(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dFCDeep":
        return AutoEncoder3D_FCDeep(latent_dim=latent_dimensions, dropout_rate=dropout_rate)

    elif model_name == "AE3dConv":
        return AutoEncoder3D_Conv(latent_dim=latent_dimensions)

    elif model_name == "AE3dLinear":
        return AutoEncoder3D_Linear(latent_dim=latent_dimensions)
        
    else:
        raise ValueError(f"Unknown model_name: {model_name}")