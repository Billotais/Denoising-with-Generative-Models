import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
import torchaudio
from torch.autograd.variable import Variable


# Helper function used in the parser to handle boolean values
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def pixel_shuffle_1d(x, upscale_factor):
    """Does the subpixel operation
    taken from https://github.com/jjery2243542/voice_conversion/blob/master/model.py"""
    batch_size, channels, steps = x.size()
    channels //= upscale_factor
    input_view = x.contiguous().view(batch_size, channels, upscale_factor, steps)
    shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
    return shuffle_out.view(batch_size, channels, steps * upscale_factor)

def get_sizes_for_layers(B):
    """Return filter sizes and number of filters for each layer given the 
    total number of filter"""
    n_channels = []
    size_filters = []
    for b in range(1, B+1):
        n_channels.append(min(2**(6 + b), 512)) # They wrote max in paper, but min in code
        size_filters.append(max(2**(7-b) + 1, 9)) # They wrote min in paper, but max in code
    
    return n_channels, size_filters


# The input channel count is equal to the the output channel count of the previous layer
# Input will be all the channel counts, shifted to the right with a 1 before

# The first argument (i.e. the number of channels at the first layer)
# is either 1 or 2 depending on if the conditional discriminator is used or not
def args_down(n_channels, size_filters, cond=False):
    """Generate an array with the arguments given to each layer for each creation\\
       Downsampling layers"""
    
    return zip([2 if cond else 1] + n_channels[:-1], n_channels, size_filters)

# Input filter count is the size of the bottlneck for the first up layer
# And then it will be the count of the previous up layer, which is equal to twice the count of the down layer
# (since we do some stacking with the skip connections)

# Output filter count  will be twice the count of the down layer 
# so that after the subpixel we get the same count as in the down layer
# and we can stack them together
def args_up(n_channels, size_filters):
    """Generate an array with the arguments given to each layer for each creation\\
       Upsampling layers"""
    return zip([int(n_channels[-1]/2)] + n_channels[::-1][:-1], n_channels[::-1], size_filters[::-1])


# unfold dimension to make our sliding window
def sliding_window(x, window_size, step_size=1):
    return x.unfold(0,window_size,step_size)



def ones_target(size):
    """Tensor containing ones, with shape = size"""
    
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    """Tensor containing zeros, with shape = size"""
    
    data = Variable(torch.zeros(size, 1))
    return data

# Concatenate a list of tensor along the time dimnesion
def concat_list_tensors(tensor_list):
    out = torch.cat(tuple(tensor_list), 2)
    return out

# Concatenate a list of tensor along the time dimnesion
# but cut them before that since we have sopme overlap
# used for audio reconstruction to avoid artefacts at joining points
def cut_and_concat_tensors(tensor_list, window, stride):
    # first cut the tensors "in the middle", and concat them together
    limit = int((window - stride) / 2)
    
    cut_tensors = []
    
    for tensor in tensor_list[1:-1]:
        cut_tensors.append(tensor[:,:,limit:-limit])

    concat = concat_list_tensors(cut_tensors)

    # then cut the 2 tensor on the borders, and concatenate them aswell
    concat = torch.cat((tensor_list[0][:,:,:-limit], concat), 2)
    concat = torch.cat((concat, tensor_list[-1][:,:,limit:]), 2)
    return concat

def create_output_audio(outputs, rate, name, window, stride, batch):
    # Regroup all elements of a batchsize=b into a list of tensor of batchsize=1
    outputs = [torch.chunk(x, batch, dim=0) for x in outputs]
    outputs = [val for sublist in outputs for val in sublist]
    # Concatenate them
    out = cut_and_concat_tensors(outputs, window, stride)
    # Remove unnecessary dimensions
    out_formated = out.reshape((1, out.size()[2]))
    # Create a wav file from it
    torchaudio.save("out/"+name+"/out.wav", out_formated, rate, precision=16, channels_first=True)

def plot(loss_train, loss_test, loss_train_gan, loss_test_gan, loss_normal, loss_train_ae, loss_test_ae, name, GAN, AE):


    mpl.style.use('seaborn')
    
    # Plot generator loss
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    fig.suptitle('Loss Generator')
   
    ax1.plot(loss_train, label='Train loss', color='b')
    ax1.plot(loss_test, label='Test loss', color='r')
    ax1.set_yscale('log')
    ax1.set_xlabel('100*batch')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(loss_train, label='Train loss', color='b')
    ax2.plot(loss_normal, label='Train loss normal', color='g')
    ax2.set_yscale('log')
    ax2.set_xlabel('100*batch')
    ax2.set_ylabel('loss')
    ax2.legend()

    ax3.plot(loss_test, label='Test loss', color='r')
    ax3.set_yscale('log')
    ax3.set_xlabel('100*batch')
    ax3.set_ylabel('loss')
    ax3.legend()

    fig.savefig('out/'+name+'/loss.png', bbox_inches='tight')
    fig.clf()
    plt.close()

    if (GAN):
        # Plot discriminator loss
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
        fig.suptitle('Loss Discriminator')
    
        ax1.plot(loss_train_gan, label='Train loss', color='b')
        ax1.plot(loss_test_gan, label='Test loss', color='r')
        ax1.set_yscale('log')
        ax1.set_xlabel('100*batch')
        ax1.set_ylabel('loss')
        ax1.legend()

        ax2.plot(loss_train_gan, label='Train loss', color='b')
        ax2.set_yscale('log')
        ax2.set_xlabel('100*batch')
        ax2.set_ylabel('loss')
        ax2.legend()

        ax3.plot(loss_test_gan, label='Test loss', color='r')
        ax3.set_yscale('log')
        ax3.set_xlabel('100*batch')
        ax3.set_ylabel('loss')
        ax3.legend()

        fig.savefig('out/'+name+'/loss_gan.png', bbox_inches='tight')
        fig.clf()
        plt.close()
    if (AE):
        # Plot discriminator loss
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
        fig.suptitle('Loss AutoEncoder')
    
        ax1.plot(loss_train_ae, label='Train loss', color='b')
        ax1.plot(loss_test_ae, label='Test loss', color='r')
        ax1.set_yscale('log')
        ax1.set_xlabel('100*batch')
        ax1.set_ylabel('loss')
        ax1.legend()

        ax2.plot(loss_train_ae, label='Train loss', color='b')
        ax2.set_yscale('log')
        ax2.set_xlabel('100*batch')
        ax2.set_ylabel('loss')
        ax2.legend()

        ax3.plot(loss_test_ae, label='Test loss', color='r')
        ax3.set_yscale('log')
        ax3.set_xlabel('100*batch')
        ax3.set_ylabel('loss')
        ax3.legend()

        fig.savefig('out/'+name+'/loss_ae.png', bbox_inches='tight')
        fig.clf()
        plt.close()

# Algorthm 1 of CGAN, at the last level
def collaborative_sampling(generator, discriminator, x, loss, N, K):
    i = 0
    misclassified = True
    layer = generator.B-2

    # Get our x_l and yhat, use one before last layer of generator
    generator.train()
    discriminator.train()

    # First pass has arg xl=None => return xl
    xl = generator(x, lastskip=True, collab_layer=layer, xl=None)
    #xl.requires_grad=True
    # Second pass we give xl as an arg meaning that we continue from the layer of xl
    yhat = generator(x, lastskip=True, collab_layer=layer, xl=xl)


    while misclassified and i < K:
        i+=1

        # We look at what the discriminator tells us
        pred = discriminator(yhat)
        mean = pred.mean()
        if mean < 0.5: # our sample is missclassified, we have to improve it

            # Get the gradiant of the discr loss wrt xl
            loss_d = loss(pred, ones_target(N))
            loss_d.backward(retain_graph=True)
            grad = generator.get_activations_gradient()
            # Update xl to an improved value
            xl = xl - 0.1*grad
            # Get our new output from the generator
            yhat = generator(x, lastskip=True, collab_layer=layer, xl=xl)
        else: misclassified = False


    return yhat
