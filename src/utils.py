import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.variable import Variable


def pixel_shuffle_1d(x, upscale_factor):
    """Does the subpixel operation"""
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
    #test = []
    for b in range(1, B+1):
        n_channels.append(min(2**(6 + b), 512)) # They wrote max in paper, but min in code
        size_filters.append(max(2**(7-b) + 1, 9)) # They wrote min in paper, but max in code
        #test.append(min(2**(7+(B-b+1)), 512))
    
    return n_channels, size_filters


# The input channel count is equal to the the output channel count of the previous layer
# Input will be all the channel counts, shifted to the right with a 1 before
def args_down(n_channels, size_filters):
    """Generate an array with the arguments given to each layer for each creation\\
       Downsampling layers"""
    return zip([1] + n_channels[:-1], n_channels, size_filters)

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



def sliding_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    return x.unfold(0,window_size,step_size)


""" def make_train_step(model, loss_fn, optimizer):

    # Builds function that performs a step in the train loop
    def train_step(x, y):
        print(x.size())
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(y, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    
    # Returns the function that will be called inside the train loop
    return train_step """

def make_train_step_gan(generator, discriminator, loss, lambda_d, optimizer_g, optimizer_d, GAN):

    def train_step(x, y):
        N = y.size(0)
        loss = nn.BCELoss()

        #################
        # Train generator
        #################
        optimizer_g.zero_grad()
        generator.train()

        # Generate output  distriminator
        yhat = generator(x)


        # Compute the normal loss
        loss_g_normal = loss(y, yhat) 
        # Compute the adversarial loss (want want to generate realistic data)
        loss_g_adv = 0
        if GAN:
            prediction = discriminator(yhat)
            loss_g_adv = loss(prediction, ones_target(N)) 
        # Compute the global loss
        loss_g = loss_g_normal + lambda_d*loss_g_adv

        # Propagate
        loss_g.backward()
        optimizer_g.step()

        #####################
        # Train discriminator
        #####################
        if GAN:
            optimizer_d.zero_grad()
            discriminator.train()
            
            # Train with real data
            prediction_real = discriminator(y)
            loss_d_real = loss(prediction_real, ones_target(N))
            loss_d_real.backward()

            # Train with fake data
            prediction_fake = discriminator(yhat)
            loss_d_fake = loss(prediction_fake, zeros_target(N))
            loss_d_fake.backward()

            # Propagate
            optimizer_d.step()



""" def make_test_step(model, loss_fn):

    def test_step(x, y):
        
        model.eval()
        yhat = model(x)

        loss = loss_fn(y, yhat)

        return loss.item(), yhat

    return test_step """

def make_test_step_gan(generator, discriminator, loss_fn, GAN):

    def test_step(x, y):
        N = y.size(0)
        loss = nn.BCELoss()

        # Loss of G
        generator.eval()
        yhat = generator(x)
        loss_g = loss_fn(y, yhat).item()

        # Loss of D
        loss_d = 0
        if GAN:
            pred = discriminator(yhat)
            loss_d = loss(pred, zeros_target(N)).item()

        return loss_g, loss_d, yhat

    return test_step

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

def concat_list_tensors(tensor_list):
    out = torch.cat(tuple(tensor_list), 2)
    return out

def cut_and_concat_tensors(tensor_list, window, stride):
    limit = int((window - stride) / 2)
    
    cut_tensors = []
    
    for tensor in tensor_list[1:-1]:
        cut_tensors.append(tensor[:,:,limit:-limit])



    concat = concat_list_tensors(cut_tensors)
    concat = torch.cat((tensor_list[0][:,:,:-limit], concat), 2)
    concat = torch.cat((concat, tensor_list[-1][:,:,limit:]), 2)
    return concat
