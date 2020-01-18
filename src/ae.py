# Main file for the autoencoder

from datasets import IDDataset
from files import MAESTROFiles, SimpleFiles
from network import (AutoEncoder)
from torch.utils.data import ConcatDataset, DataLoader
import torch.optim as optim
from utils import (concat_list_tensors, create_output_audio,
                   cut_and_concat_tensors, str2bool)

import os
import argparse
import torch
import torch.nn as nn
import sys

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np



# Type settings depending on the device used
torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')  

def init():

    # Define all the argumements
    ap = argparse.ArgumentParser()
   
    ap.add_argument("-c", "--count", required=False, help="number of mini-batches per epoch [int], default=-1 (use all data) ", type=int, default=-1)
    ap.add_argument("-o", "--out", required=False, help="number of samples for the output file [int], default=500", type=int, default=500)
    ap.add_argument("-e", "--epochs", help="number of epochs [int], default=10", type=int, default=10)
    ap.add_argument("-b", "--batch", help="size of a minibatch [int], default=32", type=int, default=32)
    ap.add_argument("-w", "--window", help="size of the sliding window [int], default=2048", type=int, default=2048)
    ap.add_argument("-s", "--stride", help="stride of the sliding window [int], default=1024", type=int, default=1024)
    ap.add_argument("-d", "--depth", help="number of layers of the network [int], default=4, maximum allowed is log2(window)-1", type=int, default=4)
    ap.add_argument("-n", "--name",  required=True, help="name of the folder in which we want to save data for this model [string], mandatory", type=str)
    ap.add_argument("--dropout", help="value for the dropout used the network [float], default=0.5", type=float, default=0.5)
    ap.add_argument("--train_n", help="number of songs used to train [int], default=-1 (use all songs)", type=int, default=-1)
    ap.add_argument("--dataset", help="type of the dataset[simple|type], where 'type' is a custom dataset type implemented in load_data(), default=simple", type=str, default="simple")
    ap.add_argument("--data_root", help="root of the dataset [path], default=/data/lois-data/models/maestro", type=str, default="/data/lois-data/models/maestro/")
    ap.add_argument("--rate", required=True, help="Sample rate of the output file [int], mandatory", type=int)
    ap.add_argument("--loss", required=False, help="Choose the loss for the generator, [L1, L2], default='L2')", type=str, default="L2")
    ap.add_argument("--lr_ae", required=False, help="learning rate for the autoencoder [float], default=0.0001", type=float, default=0.0001)
    ap.add_argument("--scheduler", required=False, help="enable the scheduler [bool], default=False", type=str2bool, default=False)

    args = ap.parse_args()
    variables = vars(args)

    #print(variables)

    global ROOT
    ROOT = variables['data_root']
    
    count = variables['count']
    out = variables['out']
    epochs = variables['epochs']
    batch = variables['batch']
    window = variables['window']
    stride = variables['stride']
    depth = variables['depth']
    train_n = variables['train_n']
    dataset = variables['dataset']
    rate = variables['rate']
    loss = variables['loss']
    name = variables['name']
    dropout = variables['dropout']
    lr_ae = variables['lr_ae']
    scheduler = variables['scheduler']
    

    os.system("rm -rf out/" + name)
    os.system("mkdir out/" + name)
    os.system("mkdir out/" + name + "/models")
    os.system("mkdir out/" + name + "/tmp")
    os.system("mkdir out/" + name + "/losses")

    # Save the command in a file for future reference
    with open("out/" + name + "/command", "w") as text_file:
        text_file.write(" ".join(sys.argv))
    
    pipeline(depth, train_n, dataset, rate, batch, window, stride, name, lr_ae, epochs, count, loss, scheduler, dropout, out)

# Initialize all the networks
def init_net(depth, dropout):
    
    ae = AutoEncoder(depth, dropout, verbose=0)
    
    # Put them on cuda if available
    if torch.cuda.is_available():
        ae.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using : " + str(device))
    print("Network initialized\n")

    return ae, device

def load_data(train_n, val_n, dataset, rate, batch_size, window, stride, run_name):
    
    # Load the correct file structure according to the parameter
    f = None
    if dataset == 'simple': f = SimpleFiles(ROOT, 0.9)
    else: 
        print("unknow dataset type")
        exit()

    # Get the names of the file for each category
    names_train = f.get_train(train_n)
    print("Train files : " + str(names_train))
    names_val = f.get_val()
    print("\nVal files : " + str(names_val))
    names_test = f.get_test()
    print("\nTest files : " + str(names_test) + "\n")

    # Create a dataset Object for each category, using the desired preprocessing piepline
    datasets_train = [IDDataset(run_name, n, window, stride, rate) for n in names_train]
    datasets_test =  [IDDataset(run_name, n, window, stride, rate, test=True) for n in names_test]
    datasets_val =   [IDDataset(run_name, n, window, stride, rate, size=128, start=32) for n in names_val]

    # Since those are actually lists of datasets (one for each file), concatenate them
    data_train = ConcatDataset(datasets_train)
    data_test = ConcatDataset(datasets_test)
    data_val = ConcatDataset(datasets_val)
    
    # Put the data into dataloaders with the correct batch size, and with some shuffle
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    print("Data loaded\n")
    return train_loader, test_loader, val_loader


def pipeline(depth, train_n, dataset, rate, batch, window, stride, name, lr_ae, epochs, count, loss, scheduler, dropout, out):
    # Init net and cuda
    ae, device = init_net(depth=depth, dropout=dropout)
    
    # Open data, split train and val set, depending on if we want to do a full training or only the evaluation
    train_loader, test_loader, val_loader = load_data(train_n=train_n, val_n=1, dataset=dataset, rate=rate, batch_size=batch, window=window, stride=stride, run_name=name)


    # Create our optimizers
    adam_ae = optim.Adam(ae.parameters(), lr=lr_ae)

    train(ae=ae, loader=train_loader, val=val_loader, epochs=epochs, count=count, name=name, loss=loss,  optim_ae=adam_ae, device=device, scheduler=scheduler)

    # Once it's trained, generate an improved audio by calling test
    outputs = test(ae=ae, loader=test_loader, count=out, name=name, loss=loss, device=device)

    # Put the data together into a real file
    create_output_audio(outputs = outputs, rate=rate, name=name, window = window, stride=stride, batch=batch)
    print("Output file created")

    os.system("rm out/" + name + "/tmp -rf")

    os.system("python metrics.py --source out/" + name +"/in.wav --generated out/" + name +"/out.wav --target out/" + name + "/target.wav --count " + str(200000) + " > out/" + name + "/metrics")
    print("Metrics computed, all done !")

def train(ae, loader, val, epochs, count, name, loss,  optim_ae, device, scheduler):

    print("Training for " + str(epochs) +  " epochs, " + str(count) + " mini-batches per epoch")
    
    loss = nn.MSELoss() if loss == "L2" else nn.L1Loss()

    # create the functions used at each step of the training / testing process
    train_step = make_train_step(ae, loss, optim_ae)
    test_step = make_test_step(ae, loss)

    # Initialize all the buffers to save the losses
    losses_ae, val_losses_ae = [],[]
    loss_buffer_ae = []
    
    # Initialize our two scheduler, one for the generator and one for the discriminator
    scheduler_ae = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_ae, factor=0.5, threshold=1e-2, cooldown=0, verbose=True)
   


    for epoch in range(1, epochs+1):

        # correct the count variable if needed (the argument)
        total = len(loader)
        if (total < count or count < 0): 
            count = total 
        
        curr_count = 0

        for x_batch, y_batch in loader:

            # Put the networks and the data on the correct device
            ae.to(device)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Train using the current batch, and save the losses
            loss_ae = train_step(x_batch, y_batch)


            
            loss_buffer_ae.append(loss_ae) # Autoencoder loss

            # Stop if count reached
            curr_count += 1
            if (curr_count >= count): break

            # If 10 batches done, compute the averages over the last 10 batches for some graphs
            if (len(loss_buffer_ae) % 100 == 0):
                              

                # Get average train loss
                losses_ae.append(sum(loss_buffer_ae)/len(loss_buffer_ae))

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [AE loss: %f] "
                    % (epoch, epochs, curr_count, count, losses_ae[-1]))
                loss_buffer_ae = []

                # Compute average validation loss
                
                val_loss_buffer_ae = []

                # For each sample in the validation data
                for x_val, y_val in val:

                    ae.to(device)
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    # COmpute and save the losses
                    loss_ae, _ = test_step(x_val, y_val)
                    
                    val_loss_buffer_ae.append(loss_ae)
        

                # Compute and store the average val losses
                
                val_losses_ae.append(sum(val_loss_buffer_ae)/len(val_loss_buffer_ae))

                # Every 500 batches, plot and decrease lr if needed with the scheduler
                if (len(losses_ae) % 5 == 0):
                    plot(losses_ae, val_losses_ae, name)
                    if scheduler:
                        scheduler_ae.step(val_losses_ae[-1])
      

        # Save the model for the epoch
        torch.save({
            'ae_state_dict': ae.state_dict(),
            'optim_ae_state_dict': optim_ae.state_dict(),
            }, "out/" + name + "/models/model_" + str(epoch) + ".tar")

        # Save the losses
        np.save('out/' + name + '/losses/loss_train_ae.npy', np.array(losses_ae))
        np.save('out/' + name + '/losses/loss_test_ae.npy', np.array(val_losses_ae))


    # Do the final update on the graph
    plot(losses_ae, val_losses_ae, name)

def make_train_step(ae, loss, optimizer_ae):

    def train_step(x, y):


        optimizer_ae.zero_grad()
        ae.train()

        # Make prediction and compute classical L2 loss on the output
        _, prediction_ae = ae(y)
        loss_ae = loss(prediction_ae, y)

        # Propagate
        loss_ae.backward()
        loss_ae = loss_ae.item()

        optimizer_ae.step()

        # Return the loss
        return loss_ae.item()

    return train_step


def test(ae, loader, count, name, loss, device):

    loss = nn.MSELoss() if loss == "L2" else nn.L1Loss()

    # Function used as a test step created
    test_step = make_test_step(ae, loss)

    # Test model
    losses = []
    outputs = []

    # We want to disabled gradiants since we won't do any update
    #with torch.no_grad():
    for x_test, y_test in loader:

        ae.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        # Get the loss, and the generated sample, and save them
        loss, y_test_hat = test_step(x_test, y_test)

        losses.append(loss)
        outputs.append(y_test_hat.to('cpu'))

        # stop if enough samples generated
        if (count > 0 and len(losses) >= count ): break

    # Plot the losses
    plt.plot(losses)
    plt.yscale('log')
    plt.savefig('out/'+name+'/test.png')
    plt.clf()
    return outputs

def make_test_step(ae, loss_fn):

    def test_step(x, y):
        
        _, pred = ae(x)
        loss_ae = loss_fn(pred, y)
        return loss_ae.item()

    return test_step


def plot(loss_train, loss_test, name):


    mpl.style.use('seaborn')
    
    # Plot generator loss
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20,10))
    fig.suptitle('Loss Generator')
   
    ax1.plot(loss_train, label='Train loss', color='b')
    ax1.plot(loss_test, label='Test loss', color='r')
    ax1.set_yscale('log')
    ax2.set_xlabel('100*batch')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(loss_train, label='Train loss', color='b')
    ax2.set_yscale('log')
    ax2.set_xlabel('100*batch')
    ax2.set_ylabel('loss')
    ax2.legend()

    ax3.plot(loss_test, label='Test loss', color='r')
    ax3.set_yscale('log')
    ax2.set_xlabel('100*batch')
    ax3.set_ylabel('loss')
    ax3.legend()

    fig.savefig('out/'+name+'/loss.png', bbox_inches='tight')
    fig.clf()
    plt.close()


if __name__ == "__main__":
    init()
