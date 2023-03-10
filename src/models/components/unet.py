from torch import nn, concat, rand
import torch

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.convBlock(x)
    

class Down(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)
    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.convBlock = DoubleConvBlock(in_channels, out_channels)

    def forward(self, copy, x):
        x = self.up(x)
        x = concat([x, copy], dim=1)
        return self.convBlock(x)
             
class Unet(nn.Module):
    def __init__(self, in_channels, n_classes=2, channels = [64, 128, 256, 512, 1024]) -> None:
        super().__init__()
        self.inl = DoubleConvBlock(in_channels, channels[0])

        self.down = nn.ModuleList([Down(channels[i], channels[i+1]) for i in range(len(channels)-1)])

        channels = channels[::-1]
        self.up =  nn.ModuleList([Up(channels[i], channels[i+1]) for i in range(len(channels)-1)])

        self.outl = nn.Conv2d(channels[-1], n_classes, kernel_size=1)

    def forward(self, x):
        x = self.inl(x)
        copy = [x]
        for layer in self.down:
            x = layer(x)  
            copy.insert(0, x)
        for i, layer in enumerate(self.up):
            x = layer(copy[i+1], x)
        x = self.outl(x)
        return x

if __name__ == '__main__':

    unet = Unet(3)
    x = rand(1, 3, 560, 560)
    
    if torch.cuda.is_available():
        print (torch.cuda.is_available())
        unet.to('cuda')
        x= x.to('cuda')
        
    m = unet(x)

    print(m.shape)


    # x = rand(16, 64, 20,20)
    # conv = DoubleConvBlock(64, 128)
    # x_conv = conv(x)
    # print('double conv:',x_conv.shape)

    # down = Down(128, 256)
    # x_down = down(x_conv)
    # print('down: ', x_down.shape) 

    # up = Up(256, 128)
    
    # x_up = up(x_conv,x_down)
    # print('up:', x_up.shape)
