
#!/bin/python

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from progress.bar import Bar
from torch.utils.data import ConcatDataset, DataLoader

from datasets import AudioUpScalingDataset, AudioWhiteNoiseDataset
from files import Files
from network import Net
from utils import (concat_list_tensors, cut_and_concat_tensors, make_test_step,
                   make_train_step)

ROOT = "/mnt/Data/maestro-v2.0.0/"
def init():

    os.system('mkdir /tmp/vita')

    ap = argparse.ArgumentParser()

    subparsers = ap.add_subparsers(title="commands", dest="command")

    sr_parser = subparsers.add_parser('super-resolution', help='"super-resolution" help')
   
    sr_parser.add_argument("-c", "--count", required=True, help="count : number of samples used for training", type=int)
    sr_parser.add_argument("-e", "--epochs", help="epochs : number of epochs", type=int, default=1)
    sr_parser.add_argument("-b", "--batch", help="batch : size of a training batch", type=int, default=16)
    sr_parser.add_argument("-w", "--window", help="window : size of the sliding window", type=int, default=1024)
    sr_parser.add_argument("-s", "--stride", help="stride : stride of the sliding window", type=int, default=512)
    sr_parser.add_argument("-d", "--depth", help="depth : number of layers of the network", type=int, default=4)

    sr_parser.add_argument("--in_rate",  required=True, help="in_rate : input_rate", type=int)
    sr_parser.add_argument("--out_rate",  required=True, help="out_rate : output_rate", type=int)

    sr_parser.add_argument("--load",  required=False, help="load already trained model ? 0 or 1", type=int, default=0)



    dn_parser = subparsers.add_parser('denoising', help='"denoising" help')
   
    dn_parser.add_argument("-c", "--count", required=True, help="count : number of samples used for training", type=int)
    dn_parser.add_argument("-e", "--epochs", help="epochs : number of epochs", type=int, default=1)
    dn_parser.add_argument("-b", "--batch", help="batch : size of a training batch", type=int, default=16)
    dn_parser.add_argument("-w", "--window", help="window : size of the sliding window", type=int, default=1024)
    dn_parser.add_argument("-s", "--stride", help="stride : stride of the sliding window", type=int, default=512)
    dn_parser.add_argument("-d", "--depth", help="depth : number of layers of the network", type=int, default=4)

    dn_parser.add_argument("-r", "--rate", help="rate : rample rate", type=int, default=10000)

    dn_parser.add_argument("--load",  required=False, help="load already trained model ? 0 or 1", type=int, default=0)

    ofdn_parser = subparsers.add_parser('overfit-dn', help='"overfit" help')
    ofsr_parser = subparsers.add_parser('overfit-sr', help='"overfit" help')
    args = ap.parse_args()
    if (args.command == 'overfit-dn'):
        return overfit_dn()
    elif (args.command == 'overfit-sr'):
        return overfit_sr()


    variables = vars(args)
    count = variables['count']
    epochs = variables['epochs']
    batch = variables['batch']
    window = variables['window']
    stride = variables['stride']
    depth = variables['depth']
    load = variables['load']

    if (args.command == 'denoising'):
        rate = variables['rate']
        denoising(count, epochs, batch, window, stride, depth, rate, load, 'denoising')

    elif (args.command == 'super-resolution'):
        in_rate = variables['in_rate']
        out_rate = variables['out_rate']
        super_resolution(count, epochs, batch, window, stride, depth, in_rate, out_rate, load, 'super-resolution')

    else:
        print("invalid argument for the model")






def init_net(depth):
    net = Net(depth, verbose = 0)

    if torch.cuda.is_available():
        net.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using : " + str(device))
    return net, device



def upscaling_dataset(filename, window, stride, in_rate, out_rate):
    return  AudioUpScalingDataset(ROOT + filename, window=window, stride=stride, compressed_rate=in_rate, target_rate=out_rate)
def denoising_dataset(filename, window, stride, rate):
    return AudioWhiteNoiseDataset(ROOT + filename, window=window, stride=stride, rate=rate)
def get_dataset_fn(name):
    if name == 'upscaling': return upscaling_dataset
    if name == 'denoising': return denoising_dataset

def load_data(year=-1, train_n=-1, test_n=-1, val_n=-1, dataset='upscaling', batch_size=16, args=[]):
    f = Files("/mnt/Data/maestro-v2.0.0", year)
    """
    train_n : number of files used as train data
    test_n : number of files used as test data
    val_n : number of files used as val data
    """
 
    names_train = f.get_train(train_n)
    print(names_train)
    names_test = f.get_test(test_n)
    names_val = f.get_validation(val_n)
    datasets_train = [get_dataset_fn(dataset)(n, *args) for n in names_train]
    datasets_test = [get_dataset_fn(dataset)(n, *args) for n in names_test]
    datasets_val = [get_dataset_fn(dataset)(n, *args) for n in names_val]
    data_train = ConcatDataset(datasets_train)
    data_test = ConcatDataset(datasets_test)
    data_val = ConcatDataset(datasets_val)
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=1, shuffle=False)
    val_loader = DataLoader(dataset=data_val, batch_size=1, shuffle=True)
    return train_loader, test_loader, val_loader

   
def train(model, loader, epochs, count, name, loss, optim, device):

    print("Training for " + str(epochs) +  " epochs, " + str(count) + " samples per epoch")
    train_step = make_train_step(model, loss, optim)

    losses = []
    for epoch in range(epochs):
        for x_batch, y_batch in Bar('Training', suffix='%(percent)d%%').iter(loader):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)

            plt.plot(losses)
            plt.yscale('log')
            plt.savefig('img/'+name+'_train.png')
            print(len(losses))
            if (count > 0 and len(losses) >= count*(epoch+1)): break
    torch.save(model, name + ".pt")
    
def val(model, loader, count, name, loss, device):

    val_step = make_test_step(model, loss)
    losses = []
    for x_val, y_val in Bar('Validation', suffix='%(percent)d%%').iter(loader):
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        loss, y_val_hat = val_step(x_val, y_val)
        losses.append(loss)

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('img/'+name+'_val.png')
        if (count > 0 and len(losses) > count ): break
def test(model, loader, count, name, loss,  device):
    test_step = make_test_step(model, loss)

    # Test model

    losses = []
    outputs = []
    for x_test, y_test in Bar('Testing', suffix='%(percent)d%%').iter(loader):
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        loss, y_test_hat = test_step(x_test, y_test)
        losses.append(loss)

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('img/'+name+'_test.png')
   
        outputs.append(y_test_hat)
        if (count > 0 and len(losses) > count ): break
    return outputs

def create_output_audio(outputs, rate, name, window, stride):
    out = cut_and_concat_tensors(outputs, window, stride)
    out_formated = out.reshape((1, out.size()[2]))
    torchaudio.save("out/"+name+".wav", out_formated, rate, precision=16, channels_first=True)


def super_resolution(count, epochs, batch, window, stride, depth, in_rate, out_rate, load=False, name='super_resolution'):

    # Init net and cuda
    net, device = init_net(depth)
    print("Network initialized")
    # Open data, split train and val set

    train_loader, test_loader, val_loader = load_data(train_n=1, test_n=1, val_n=1, dataset='upscaling', batch_size=16, args=[window, stride, in_rate, out_rate])
    print("Data loaded")
   
    adam = optim.Adam(net.parameters(), lr=0.0001)
    if load: net = torch.load(name + ".pt")
    else: train(model=net, loader = train_loader, epochs=epochs, count=count, name=name, loss=nn.MSELoss(), optim=adam, device=device)
    print("Model trained")


    val(model=net, loader=val_loader, count=500, name="validation", loss=nn.MSELoss(), device=device)
    print("Model validated")        

    outputs = test(model=net, loader=test_loader, count=1000, name='test', loss=nn.MSELoss(), device=device)
    create_output_audio(outpus = outputs, rate=out_rate, name='out', window = window, stride=stride)
    print("Output file created")

def denoising(count, epochs, batch, window, stride, depth, rate, load=False, name='denoising'):
     # Init net and cuda
    net, device = init_net(depth)
    print("Network initialized")
    # Open data, split train and val set

    train_loader, test_loader, val_loader = load_data(train_n=1, test_n=1, val_n=1, dataset='denoising', batch_size=16,args=[window, stride, rate])
    print("Data loaded")
   
    adam = optim.Adam(net.parameters(), lr=0.0001)
    if load: net = torch.load(name + ".pt")
    else: train(model=net, loader = train_loader, epochs=epochs, count=count, name='name', loss=nn.MSELoss(), optim=adam, device=device)
    print("Model trained")


    val(model=net, loader=val_loader, count=500, name="validation", loss=nn.MSELoss(), device=device)
    print("Model validated")        

    outputs = test(model=net, loader=test_loader, count=1000, name='test', loss=nn.MSELoss(), device=device)
    create_output_audio(outpus = outputs, rate=rate, name='out', window = window, stride=stride)
    print("Output file created")


   # Trouver la source du clic-clic-clic
# Faire un entrainnement long
# voir si y'a pas de décalage entre in et out.
if __name__ == "__main__":
    init()
# Essayer d'overfit avec un petit bout de musique
# Ca devrait donner un bon résultat
# Essayer de faire l'identité sans les skip connections
# ssh nice = 19 pour pas prendre trop de cpu sur le server
# pytorch seulement choisir 1 seul gpu
# mettre le code pour qu'il puisse passer de gpu a cpu automatiquement en fonction d'ou on est

# Trouver la source du clic-clic-clic
# Faire un entrainnement long
# voir si y'a pas de décalage entre in et out.

# Faire les slides

#####################################################

# identité sans le "skip add"

# garder downsample simple jusqu'au midterm
# si ca marche pas, juste essayer avec bruit blanc

# Overfit sur un petit bout de musique (retrouver le code git) (vraiment petit, e.g 2 samples)

# entrainement long sur gpu

# tourner le code avec le dataset du paper beethvoen (ajouter classe nécessaire)
# Reéflcehir a avoir les meme résultats que le papier.
# pyaudio pour compresion mp3 ?


