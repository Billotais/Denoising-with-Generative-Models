import torch
import torch.nn as nn
import torch.nn.functional as F

def pixel_shuffle_1d(x, upscale_factor):
    batch_size, channels, steps = x.size()
    channels //= upscale_factor
    input_view = x.contiguous().view(batch_size, channels, upscale_factor, steps)
    shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
    return shuffle_out.view(batch_size, channels, steps * upscale_factor)

def get_sizes_for_layers(B):
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
    return zip([1] + n_channels[:-1], n_channels, size_filters)

# Input filter count is the size of the bottlneck for the first up layer
# And then it will be the count of the previous up layer, which is equal to twice the count of the down layer
# (since we do some stacking with the skip connections)

# Output filter count  will be twice the count of the down layer 
# so that after the subpixel we get the same count as in the down layer
# and we can stack them together
def args_up(n_channels, size_filters):
    return zip([int(n_channels[-1]/2)] + n_channels[::-1][:-1], n_channels[::-1], size_filters[::-1])



def sliding_window(x, window_size, step_size=1):
    # unfold dimension to make our rolling window
    return x.unfold(0,window_size,step_size)


def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
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
    return train_step



    #  transform_input = sox.Transformer()
    #     transform_input.downsample(factor=compressed_rate)
    #     transform_input.upsample(factor=int(compressed_rate/target_rate))
    #     transform_input.build(filename, '/tmp/vita/source.wav')
       
    #     waveform_compressed, _ = torchaudio.load('/tmp/vita/source.wav')

    #     self.x = waveform_compressed[0]
    #     self.x = sliding_window(self.x, window, stride)
    #     self.x = self.x[:samples, None, :]
        
    #     # Get the target data

    #     transform_out = sox.Transformer()
    #     transform_out.downsample(factor=target_rate)
    #     transform_out.build(filename, '/tmp/vita/target.wav')

    #     waveform_target, _ = torchaudio.load('/tmp/vita/target.wav')

    #     self.y = waveform_target[0]
    #     self.y = sliding_window(self.y, window, stride)
    #     self.y = self.y[:samples, None, :]
 