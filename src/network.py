import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils import *
DROPOUT = 0.5
RELU = 0.2


################################
# Objects used by all networks #
################################
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



##########################
# Code for the Generator #
##########################

class Downsampling_G(nn.Module):
    
    def __init__(self, in_ch, out_ch, size, verbose=0):
        super(Downsampling_G, self).__init__()
        self.verbose = verbose
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=size, stride=2, padding_mode='zeros', padding=int((size-1)/2))
        self.relu = nn.LeakyReLU(RELU)
        
        
    def forward(self, x):
        if self.verbose: print("Before conv down : " + str(x.size()))
        y = self.conv(x)
        y = self.relu(y)
        if self.verbose: print("After conv down : " + str(y.size()))
        return y


class Bottleneck_G(nn.Module):
    def __init__(self, ch, size, verbose=0):
        super(Bottleneck_G, self).__init__()
        self.verbose = verbose
        self.conv =  nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=size, stride=2, padding_mode='zeros', padding = int((size-1)/2))
        self.dropout = nn.Dropout(DROPOUT)
        self.relu = nn.LeakyReLU(RELU)
      
        
    def forward(self, x):
        if self.verbose: print("Bottleneck before: " + str(x.size()))
        y = self.conv(x)
        y = self.dropout(y)
        y = self.relu(y)
        if self.verbose: print("Bottleneck after: " + str(y.size()))
        return y



class Upsampling_G(nn.Module):
    def __init__(self, in_ch, out_ch, size, verbose=0):
        super(Upsampling_G, self).__init__()
        self.verbose = verbose
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=size, stride=1, padding_mode='zeros', padding = int((size-1)/2))
        self.dropout = nn.Dropout(p=DROPOUT)
        self.relu = nn.ReLU()
        self.subpixel = Subpixel()
        self.concat = Concat()
        
    def forward(self, x1, x2):
        if self.verbose: print("Upsampling before: " + str(x1.size()))
        y = self.conv(x1)
        y = self.dropout(y)
        y = self.relu(y)
        y = self.subpixel(y)
        y = self.concat(y, x2)
        if self.verbose: print("Upsampling after: " + str(y.size()))
        return y


class LastConv_G(nn.Module):
    def __init__(self, in_ch, size, verbose=0):
        super(LastConv_G, self).__init__()
        self.verbose = verbose
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=2, kernel_size=9, stride=1, padding_mode='zeros', padding = int((size-1)/2))
        self.subpixel = Subpixel()
        self.add = Add()
           
        
    def forward(self, x1, x2, lastskip):
        if self.verbose: print("Final before: " + str(x1.size()))
        y = self.conv(x1)
        if self.verbose: print("Final conv: " + str(y.size()))
        y = self.subpixel(y)
        if self.verbose: print("Final subpixel: " + str(y.size()))
        if lastskip: y = self.add(y, x2)
        if self.verbose: print("Final add: " + str(y.size()))
        return y



class Generator(nn.Module):

    def __init__(self, depth, dropout, verbose=0):
        super(Generator, self).__init__()

        global  DROPOUT
        DROPOUT = dropout
        self.verbose = verbose
        
        self.B = depth
        n_channels, size_filters = get_sizes_for_layers(self.B)
        
        # Downsampling

        self.down = nn.ModuleList([Downsampling_G(n_ch_in, n_ch_out, size, verbose) for n_ch_in, n_ch_out, size in args_down(n_channels, size_filters)])
            
        # Bottlneck
        self.bottleneck = Bottleneck_G(n_channels[-1], size_filters[-1], verbose)
        
        # Upsampling

        self.up = nn.ModuleList([Upsampling_G(n_ch_in*2, n_ch_out*2, size, verbose) for n_ch_in, n_ch_out, size in args_up(n_channels, size_filters)])
              
        # Final layer
        self.last = LastConv_G(n_channels[0]*2, 9, verbose)

    def forward(self, x, lastskip=True, collab_layer=-1, xl=None):

        # Downsampling
        down_out = []
        xi = x
        for i in range(len(self.down)):
            xi = self.down[i](xi)
            down_out.append(xi)
            
        # Bottleneck
        b = self.bottleneck(xi)
        
        # Upsampling, with code specific to collaborative sampling
        y = b
        for i in range(len(self.up)):
            y = self.up[i](y, down_out[-(i+1)])

            # If this is the "collaborative" layer
            if (i == self.B-1-collab_layer): 
                # first pass, we return x_l and stop 
                if xl is None: return y
                # second pass, this time we have our new x_l, and we want to get the output from it
                # So we change the data at this layer, and it will be "propagated" to the other layers
                else: y = xl
            
            
        # Final layer
        y = self.last(y, x, lastskip)
       
        return y


##############################
# Code for the Discriminator #
##############################

class Downsampling_D(nn.Module):

    
    def __init__(self, in_ch, out_ch, size, verbose=0):
        super(Downsampling_D, self).__init__()
        self.verbose = verbose
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=size, stride=2, padding_mode='zeros', padding=int((size-1)/2))
        self.batchnorm = nn.BatchNorm1d(out_ch)
        self.relu = nn.LeakyReLU(RELU)
        
        
    def forward(self, x):
        if self.verbose: print("Before conv down : " + str(x.size()))
        y = self.conv(x)
        y = self.batchnorm(y)
        y = self.relu(y)
        if self.verbose: print("After conv down : " + str(y.size()))
        return y

class Discriminator(nn.Module):
    def __init__(self, depth, dropout, input_size, verbose=0, cond=False):
        super(Discriminator, self).__init__()

        global  DROPOUT
        DROPOUT = dropout
        self.verbose = verbose
        
        B = depth
        n_channels, size_filters = get_sizes_for_layers(B)
        
        # Downsampling
        self.down = nn.ModuleList([Downsampling_D(n_ch_in, n_ch_out, size, verbose) for n_ch_in, n_ch_out, size in args_down(n_channels, size_filters, cond=cond)])
        # Flatten
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=DROPOUT)

        # Compute input size on the fly
        # We need this information as a linaer layer takes a defined input size
        # Given the input data of the network (its shape), we can compute the input size at this linear layer
        # by doing a forward pass with random data of the correct size
        
        self.flattened_input_size = self.get_flatten_features(input_size, self.down, self.flatten)
        self.linear = nn.Linear(self.flattened_input_size, 1)

        self.sigmoid = nn.Sigmoid()




    # Helper to compute the size of our input after the down and flatten layers
    def get_flatten_features(self, input_size, down, flatten):
        # Make a forward pass with random data
        f = Variable(torch.ones(1,*input_size))
        for i in range(len(self.down)):
            f = down[i](f)
        f = flatten(f)
        return int(np.prod(f.size()[1:]))

    def forward(self, x):

        # Downsampling
        y = x
        for i in range(len(self.down)):
            y = self.down[i](y)
            
        y = self.dropout(y)
        y = self.flatten(y)
        y = self.linear(y)
        y = self.sigmoid(y)

        return y

##########################################
# Code for the Conditional Discriminator #
##########################################

class ConditionalDiscriminator(nn.Module):
    def __init__(self, depth, dropout, input_size, verbose=0):
        super(ConditionalDiscriminator, self).__init__()
        # This is basically the same as the normal discriminator
        # we just concatenate our two inputs at the begining
        self.concat = Concat()
        self.discriminator = Discriminator(depth, dropout, input_size, verbose=0, cond=True)
    
    def forward(self, x, z): # D(x knowing also z) = y

        # We don't want our loss to be differentiated along z
        y = self.concat(x, z.detach())
        y = self.discriminator(y)
        return y

############################
# Code for the Autoencoder #
############################

class Upsampling_AE(nn.Module):
    def __init__(self, in_ch, out_ch, size, verbose=0):
        super(Upsampling_AE, self).__init__()
        self.verbose = verbose
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=size, stride=1, padding_mode='zeros', padding = int((size-1)/2))
        self.dropout = nn.Dropout(p=DROPOUT)
        self.relu = nn.ReLU()
        self.subpixel = Subpixel()
        
        
    def forward(self, x):
        if self.verbose: print("Upsampling before: " + str(x1.size()))
        y = self.conv(x)
        y = self.dropout(y)
        y = self.relu(y)
        y = self.subpixel(y)
        if self.verbose: print("Upsampling after: " + str(y.size()))
        return y

class LastConv_AE(nn.Module):
    def __init__(self, in_ch, size, verbose=0):
        super(LastConv_AE, self).__init__()
        self.verbose = verbose
        self.conv = nn.Conv1d(in_channels=in_ch, out_channels=2, kernel_size=9, stride=1, padding_mode='zeros', padding = int((size-1)/2))
        self.subpixel = Subpixel()
           
        
    def forward(self, x):
        if self.verbose: print("Final before: " + str(x1.size()))
        y = self.conv(x)
        if self.verbose: print("Final conv: " + str(y.size()))
        y = self.subpixel(y)
        if self.verbose: print("Final subpixel: " + str(y.size()))
        if self.verbose: print("Final add: " + str(y.size()))
        return y

class AutoEncoder(nn.Module):

    def __init__(self, depth, dropout, verbose=0):
        super(AutoEncoder, self).__init__()

        global  DROPOUT
        DROPOUT = dropout
        self.verbose = verbose
        
        B = depth
        n_channels, size_filters = get_sizes_for_layers(B)
        
        # Downsampling

        self.down = nn.ModuleList([Downsampling_G(n_ch_in, n_ch_out, size, verbose) for n_ch_in, n_ch_out, size in args_down(n_channels, size_filters)])
            
        # Bottlneck
        self.bottleneck = Bottleneck_G(n_channels[-1], size_filters[-1], verbose)
        
        # Upsampling

        self.up = nn.ModuleList([Upsampling_AE(n_ch_in*2, n_ch_out*4, size, verbose) for n_ch_in, n_ch_out, size in args_up(n_channels, size_filters)])
              
        # Final layer
        self.last = LastConv_AE(n_channels[0]*2, 9, verbose)
  

    def forward(self, x):

        # Downsampling
        xi = x
        for i in range(len(self.down)):
            xi = self.down[i](xi)
            
        # Bottleneck
        b = self.bottleneck(xi)
        
        # Upsampling
        y = b
        for i in range(len(self.up)):
            y = self.up[i](y)
            
        # Final layer
        y = self.last(y)
       
        return b, y

""" 
# net = Discriminator(4, 0.5, (1, 1024))
net = AutoEncoder(4, 0.5)
# #net = Generator(4, 0.5)
print(net)

input = torch.rand(32, 1, 1024)

print(net(input)) 
"""


    
