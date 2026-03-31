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

        self.to_latent = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),

            nn.Conv3d(128, self.latent_channels, kernel_size=4, stride=4, padding=0)
            # (128,2,8,8) -> (C,1,2,2)
        )

        # Decoder
        self.from_latent = nn.Sequential(
            nn.ConvTranspose3d(
                self.latent_channels, 128,
                kernel_size=4, stride=4, padding=0
            ),
            nn.InstanceNorm3d(128),
            nn.ReLU(inplace=True),
        )  # -> (128,2,8,8)

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
        z_tensor = self.to_latent(x)  # -> (B,C,1,2,2)
        return z_tensor

    def encode(self, x):
        z_tensor = self.encode_tensor(x)
        z = z_tensor.flatten(start_dim=1)  # -> (B, latent_dim)
        return z

    def decode_from_tensor(self, z_tensor):
        x = self.from_latent(z_tensor)
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
