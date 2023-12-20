class ConvBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, enable_relu: bool = True):
        super(ConvBlock, self).__init__()
        self.enable_relu = enable_relu
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_channel),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        
        return F.relu(y, inplace=True) if self.enable_relu else y


class ResBlock(nn.Module):
    
    def __init__(self, input_channel: int, block_base_channel: int):
        super(ResBlock, self).__init__()
        equal_channel_size = input_channel == block_base_channel * 4
    
        self.block = nn.Sequential(
            ConvBlock(input_channel, block_base_channel, 1),
            ConvBlock(block_base_channel, block_base_channel, 3, padding = 1),
            ConvBlock(block_base_channel, block_base_channel * 4, 1)
        )
        
        self.downsample = nn.Identity() if equal_channel_size else nn.Sequential(
            ConvBlock(input_channel, block_base_channel * 4, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.downsample(x)