import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

""" test model_1 """
class DropoutModel8x8(nn.Module):
    def __init__ (self, channel):
        """
        Define useful layers
        Argument:
        channel: number of channel, or depth or number of different sprite types
        """
        super(DropoutModel8x8, self).__init__()
        # expected input size: (batch_size, channel, row=8, column=8)
        self.dropout_1 = nn.Dropout2d(0.3)

        self.conv_1 = nn.Conv2d(channel, channel*2, kernel_size=3, stride=1) # 6x6
        self.conv_2 = nn.Conv2d(channel*2, channel*4, kernel_size=3, stride=1) # 4x4
        self.conv_3 = nn.Conv2d(channel*4, channel*8, kernel_size=3, stride=1) # 2x2

        self.conv_middle = nn.Conv2d(channel*8, channel*8, kernel_size=3, stride=1, padding=1) # 2x2

        self.conv_T1 = nn.ConvTranspose2d(channel*8, channel*4, kernel_size=3, stride=1) # 4x4
        self.conv_T2 = nn.ConvTranspose2d(channel*4, channel*2, kernel_size=3, stride=1) # 6x6
        self.conv_T3 = nn.ConvTranspose2d(channel*2, channel, kernel_size=3, stride=1) # 8x8


    def forward(self, x):
        if (self.training):
            x = self.dropout_1(x)
        
        x = func.relu(self.conv_1(x))
        x = func.relu(self.conv_2(x))
        x = func.relu(self.conv_3(x))

        x = self.conv_middle(x)
        
        x = self.conv_T1(x)
        x = self.conv_T2(x)
        # NOBUG
        x = torch.sigmoid(self.conv_T3(x))

        return x
