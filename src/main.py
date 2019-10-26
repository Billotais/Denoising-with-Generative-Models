
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

from datasets import AudioUpScalingDataset, AudioWhiteNoiseDataset, AudioIDDataset
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
    sr_parser.add_argument("-n", "--name",  required=True, help="name of model", type=str)



    dn_parser = subparsers.add_parser('denoising', help='"denoising" help')
   
    dn_parser.add_argument("-c", "--count", required=True, help="count : number of samples used for training", type=int)
    dn_parser.add_argument("-e", "--epochs", help="epochs : number of epochs", type=int, default=1)
    dn_parser.add_argument("-b", "--batch", help="batch : size of a training batch", type=int, default=16)
    dn_parser.add_argument("-w", "--window", help="window : size of the sliding window", type=int, default=1024)
    dn_parser.add_argument("-s", "--stride", help="stride : stride of the sliding window", type=int, default=512)
    dn_parser.add_argument("-d", "--depth", help="depth : number of layers of the network", type=int, default=4)

    dn_parser.add_argument("-r", "--rate", help="rate : rample rate", type=int, default=10000)

    dn_parser.add_argument("--load",  required=False, help="load already trained model ? 0 or 1", type=int, default=0)
    dn_parser.add_argument("-n", "--name",  required=True, help="name of model", type=str)

    ofdn_parser = subparsers.add_parser('overfit-dn', help='"overfit" help')
    ofsr_parser = subparsers.add_parser('overfit-sr', help='"overfit" help')

    id_parser = subparsers.add_parser('identity', help='"identity" help')
    id_parser.add_argument("-c", "--count", required=True, help="count : number of samples used for training", type=int)
    id_parser.add_argument("-e", "--epochs", help="epochs : number of epochs", type=int, default=1)
    id_parser.add_argument("-b", "--batch", help="batch : size of a training batch", type=int, default=16)
    id_parser.add_argument("-w", "--window", help="window : size of the sliding window", type=int, default=1024)
    id_parser.add_argument("-s", "--stride", help="stride : stride of the sliding window", type=int, default=512)
    id_parser.add_argument("-d", "--depth", help="depth : number of layers of the network", type=int, default=4)

    id_parser.add_argument("-r", "--rate", help="rate : rample rate", type=int, default=10000)

    id_parser.add_argument("--load",  required=False, help="load already trained model ? 0 or 1", type=int, default=0)
    id_parser.add_argument("-n", "--name",  required=True, help="name of model", type=str)

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
    load = True if variables['load'] == 1 else 0
    name = variables['name']

    if (args.command == 'denoising'):
        rate = variables['rate']
        denoising(count, epochs, batch, window, stride, depth, rate, load, name)

    elif (args.command == 'super-resolution'):
        in_rate = variables['in_rate']
        out_rate = variables['out_rate']
        super_resolution(count, epochs, batch, window, stride, depth, in_rate, out_rate, load, name)
    elif (args.command == 'identity'):
        rate = variables['rate']
        return identity(count, epochs, batch, window, stride, depth, rate, load=load)
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
def identity_dataset(filename, window, stride, rate):
    return AudioIDDataset(ROOT + filename, window,stride,-1)
def get_dataset_fn(name):
    if name == 'upscaling': return upscaling_dataset
    if name == 'denoising': return denoising_dataset
    if name == 'identity': return identity_dataset

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
    print(names_test)
    names_val = f.get_validation(val_n)
    print(names_val)
    datasets_train = [get_dataset_fn(dataset)(n, *args) for n in names_train]
    datasets_test = [get_dataset_fn(dataset)(n, *args) for n in names_test]
    datasets_val = [get_dataset_fn(dataset)(n, *args) for n in names_val]
    data_train = ConcatDataset(datasets_train)
    data_test = ConcatDataset(datasets_test)
    data_val = ConcatDataset(datasets_val)
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=1, shuffle=False)
    val_loader = DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, val_loader

   
def train(model, loader, epochs, count, name, loss, optim, device):

    print("Training for " + str(epochs) +  " epochs, " + str(count) + " samples per epoch")
    train_step = make_train_step(model, loss, optim)

    losses = []
    for epoch in range(epochs):
        bar = Bar('Training', max=count)
        for x_batch, y_batch in loader:

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)

            plt.plot(losses)
            plt.yscale('log')
            
            bar.next()
            if (count > 0 and len(losses) >= count*(epoch+1)): break
        bar.finish()
        plt.savefig('img/'+name+'_train.png')
    torch.save(model.state_dict(), name + ".pt")
    
    
def val(model, loader, count, name, loss, device):

    val_step = make_test_step(model, loss)
    losses = []
    bar = Bar('Validation', max=count)
    for x_val, y_val in loader:
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        loss, y_val_hat = val_step(x_val, y_val)
        losses.append(loss)

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('img/'+name+'_val.png')
        bar.next()
        if (count > 0 and len(losses) > count ): break
    bar.finish()

def test(model, loader, count, name, loss,  device):
    test_step = make_test_step(model, loss)

    # Test model

    losses = []
    outputs = []
    bar = Bar('Testing', max=count)
    for x_test, y_test in loader:
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        loss, y_test_hat = test_step(x_test, y_test)
        losses.append(loss)

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('img/'+name+'_test.png')
   
        outputs.append(y_test_hat)
        bar.next()
        if (count > 0 and len(losses) > count ): break
    bar.finish()
    return outputs

def create_output_audio(outputs, rate, name, window, stride):
    #outputs = [torch.flatten(x, 0) for x in outputs]# addedd to handle batches in output 
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
    if load: 
        net = Net(depth)
        net.load_state_dict(torch.load(name + ".pt"))
    else: train(model=net, loader = train_loader, epochs=epochs, count=count, name=name, loss=nn.MSELoss(), optim=adam, device=device)
    print("Model trained")


    val(model=net, loader=val_loader, count=500, name=name, loss=nn.MSELoss(), device=device)
    print("Model validated")        

    outputs = test(model=net, loader=test_loader, count=1000, name=name, loss=nn.MSELoss(), device=device)
    create_output_audio(outpus = outputs, rate=out_rate, name=name, window = window, stride=stride)
    print("Output file created")



def denoising(count, epochs, batch, window, stride, depth, rate, load=False, name='denoising'):
     # Init net and cuda
    net, device = init_net(depth)
    print("Network initialized")
    # Open data, split train and val set

    train_loader, test_loader, val_loader = load_data(train_n=1, test_n=1, val_n=1, dataset='denoising', batch_size=16,args=[window, stride, rate])
    print("Data loaded")
   
    adam = optim.Adam(net.parameters(), lr=0.0001)
    if load: 
        net = Net(depth)
        net.load_state_dict(torch.load(name + ".pt"))
    else: train(model=net, loader = train_loader, epochs=epochs, count=count, name=name, loss=nn.MSELoss(), optim=adam, device=device)
    print("Model trained")


    val(model=net, loader=val_loader, count=500, name=name, loss=nn.MSELoss(), device=device)
    print("Model validated")        

    outputs = test(model=net, loader=test_loader, count=1000, name=name, loss=nn.MSELoss(), device=device)
    create_output_audio(outpus = outputs, rate=rate, name=name, window = window, stride=stride)
    print("Output file created")

def identity(count, epochs, batch, window, stride, depth, rate, load=False, name='identity'):

    # Init net and cuda
    net, device = init_net(depth)
    print("Network initialized")
    # Open data, split train and val set

    train_loader, test_loader, val_loader = load_data(train_n=1, test_n=1, val_n=1, dataset='identity', batch_size=16, args=[window, stride, rate])
    print("Data loaded")
   
    adam = optim.Adam(net.parameters(), lr=0.0001)
    if load: 
        net = Net(depth)
        net.load_state_dict(torch.load(name + ".pt"))
    else: train(model=net, loader = train_loader, epochs=epochs, count=count, name=name, loss=nn.MSELoss(), optim=adam, device=device)
    print("Model trained")


    #val(model=net, loader=val_loader, count=500, name=name, loss=nn.MSELoss(), device=device)
    #print("Model validated")        

    outputs = test(model=net, loader=test_loader, count=500, name=name, loss=nn.MSELoss(), device=device)
    create_output_audio(outputs = outputs, rate=rate, name=name, window = window, stride=stride)
    print("Output file created")
    
def overfit_sr():
    FILENAME = "/mnt/Data/maestro-v2.0.0/2006/MIDI-Unprocessed_21_R1_2006_01-04_ORIG_MID--AUDIO_21_R1_2006_01_Track01_wav.wav"
    # Init net and cuda

    net, device = init_net(4)

    # Open data, split train and val set


    data = AudioUpScalingDataset(FILENAME, window=1024, stride=512, compressed_rate=5000, target_rate=10000, size=2, start=40)

    # Load data into batches

    train_loader = DataLoader(dataset=data, batch_size=16, shuffle=True)
    train_step = make_train_step(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=0.0002))
    
    test_data = AudioUpScalingDataset("MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.wav"
, window=1024, stride=512, compressed_rate=5000, target_rate=10000, size=100)

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    test_step = make_test_step(net, nn.MSELoss())

    # Train model
    n_epochs = 5000
    losses = []
    for epoch in range(n_epochs):
        for x_batch, y_batch in Bar('Training - epoch ' + str(epoch), suffix='%(percent)d%%').iter(train_loader):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('img/overfit_sr_train.png')
        




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

####################################################

# Augmenter le learning rate de Adam (utiliser celui pas default)
# Experience avec 2 samples, comparer selon les learning rates.
# Essayer de reproduire les reulstats du papier (meme valuers de taille, epochs, reslution, etc)
# Implementer les metrics du paper
# sauvegearde model avec les parameetres, pas trop souvent
# quand on load on model ajouter la possibilité de continuer e train.





