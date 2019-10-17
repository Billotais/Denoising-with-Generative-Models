import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

class Subpixel(nn.Module):
    """Subpixel module"""
    def __init__(self):
        super(Subpixel, self).__init__()
           
    def forward(self, x):
        """Interleave two layers into only one of hoigher dimension"""
        y = pixel_shuffle_1d(x, 2)
        return y



class Concat(nn.Module):
    """Concat module"""
    def __init__(self):
        super(Concat, self).__init__()
           
        
    def forward(self, x1, x2):
        
        y = torch.cat((x1, x2), 1) # concat on dim 1 (channel dimension)
        return y


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
           
        
    def forward(self, x1, x2):
        y = torch.add(x1, x2)
        return y



class Downsampling(nn.Module):
    
    def __init__(self, in_ch, out_ch, size, verbose=0):
        super(Downsampling, self).__init__()
        self.verbose = verbose
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=size, stride=2, padding_mode='zeros', padding=int((size-1)/2))
        self.relu = nn.LeakyReLU(0.2)
        
        
    def forward(self, x, training):
        if self.verbose: print("Before conv down : " + str(x.size()))
        y = self.conv(x)
        y = self.relu(y)
        if self.verbose: print("After conv down : " + str(y.size()))
        return y




class Bottleneck(nn.Module):
    def __init__(self, ch, size, verbose=0):
        super(Bottleneck, self).__init__()
        self.verbose = verbose
        self.conv =  nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=size, stride=2, padding_mode='zeros', padding = int((size-1)/2))
        self.dropout = nn.Dropout(0.5, )
        self.relu = nn.LeakyReLU(0.2)
      
        
    def forward(self, x, training):
        if self.verbose: print("Bottleneck before: " + str(x.size()))
        y = self.conv(x)
        if training: y = self.dropout(y)
        y = self.relu(y)
        if self.verbose: print("Bottleneck after: " + str(y.size()))
        return y



class Upsampling(nn.Module):
    def __init__(self, in_ch, out_ch, size, verbose=0):
        super(Upsampling, self).__init__()
        self.verbose = verbose
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=size, stride=1, padding_mode='zeros', padding = int((size-1)/2))
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.subpixel = Subpixel()
        self.concat = Concat()
        
    def forward(self, x1, x2, training):
        if self.verbose: print("Upsampling before: " + str(x1.size()))
        y = self.conv(x1)
        if training: y = self.dropout(y)
        y = self.relu(y)
        y = self.subpixel(y)
        y = self.concat(y, x2)
        if self.verbose: print("Upsampling after: " + str(y.size()))
        return y



class LastConv(nn.Module):
    def __init__(self, in_ch, size, verbose=0):
        super(LastConv, self).__init__()
        self.verbose = verbose
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=2, kernel_size=9, stride=1, padding_mode='zeros', padding = int((size-1)/2))
        self.subpixel = Subpixel()
        self.add = Add()
           
        
    def forward(self, x1, x2, training):
        if self.verbose: print("Final before: " + str(x1.size()))
        y = self.conv(x1)
        if self.verbose: print("Final conv: " + str(y.size()))
        y = self.subpixel(y)
        if self.verbose: print("Final subpixel: " + str(y.size()))
        y = self.add(y, x2)
        if self.verbose: print("Final add: " + str(y.size()))
        return y



class Net(nn.Module):

    def __init__(self, depth, verbose=0):
        super(Net, self).__init__()

        
        self.verbose = verbose
        
        B = depth
        n_channels, size_filters = get_sizes_for_layers(B)
        
        # Downsampling
        self.down = []
        for n_ch_in, n_ch_out, size in args_down(n_channels, size_filters):
            self.down.append(Downsampling(n_ch_in, n_ch_out, size, verbose))
            
        # Bottlneck
        self.bottleneck = Bottleneck(n_channels[-1], size_filters[-1], verbose)
        
        # Upsampling
        self.up = []
        for n_ch_in, n_ch_out, size in args_up(n_channels, size_filters):
            self.up.append(Upsampling(n_ch_in*2, n_ch_out*2, size, verbose))
              
        # Final layer
        self.last = LastConv(n_channels[0]*2, 9, verbose)
        
        
        

    def forward(self, x):

        # Since the network is not able to automaticaly propagate the
        # "training" variable down the modules (probably because we put the modules in a list)
        # I added the "training" argument to the forward function.
        
        # Downsampling
        down_out = []
        xi = x
        for i in range(len(self.down)):
            xi = self.down[i](xi, self.training)
            down_out.append(xi)
            
        # Bottleneck
        b = self.bottleneck(xi, self.training)
        
        # Upsampling
        y = b
        for i in range(len(self.up)):
            y = self.up[i](y, down_out[-(i+1)], self.training)
            
        # Final layer
        y = self.last(y, x, self.training)
       
        return y
    