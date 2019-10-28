
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
import progressbar
from torch.utils.data import ConcatDataset, DataLoader

from datasets import AudioUpScalingDataset, AudioWhiteNoiseDataset, AudioIDDataset
from files import MAESTROFiles, SimpleFiles
from network import Net
from utils import (concat_list_tensors, cut_and_concat_tensors, make_test_step,
                   make_train_step)

ROOT = "/mnt/Data/maestro-v2.0.0/"
ROOT = "/mnt/Data/Beethoven/"
#ROOT = "/data/lois-data/Beethoven/"

torch.set_default_tensor_type('torch.FloatTensor')
#torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Uncomment this to run on GPU
def init():

    #os.system('mkdir /tmp/vita')

    ap = argparse.ArgumentParser()

   
    ap.add_argument("-c", "--count", required=True, help="number of mini-batches used for training", type=int)
    ap.add_argument("-o", "--out", required=False, help="number of samples to output", type=int, default=500)
    ap.add_argument("-e", "--epochs", help="number of epochs, default 1", type=int, default=1)
    ap.add_argument("-b", "--batch", help="size of a training batch, default 32", type=int, default=32)
    ap.add_argument("-w", "--window", help="size of the sliding window, default 2048", type=int, default=2048)
    ap.add_argument("-s", "--stride", help="stride of the sliding window, default 1024", type=int, default=1024)
    ap.add_argument("-d", "--depth", help="number of layers of the network, default 4", type=int, default=4)
    ap.add_argument("--train_n", help="number of songs used to train, default 1", type=int, default=1)
    ap.add_argument("--test_n", help="number of songs used to test, default 1", type=int, default=1)
    ap.add_argument("--load",  required=False, help="load already trained model to evaluate ? 0 or 1, default 0", type=int, default=0)
    ap.add_argument("--continue",  required=False, help="load already trained model to continue training ? 0 or 1, default 0", type=int, default=0)
    ap.add_argument("--dataset", required=False, help="name of the dataset (definded in load_Data()), default beethoven", type=str, default="beethoven")
    ap.add_argument("-n", "--name",  required=True, help="name of model", type=str)

    pa = argparse.ArgumentParser()
    subparsers = pa.add_subparsers(title="commands",dest="command")


    sr_parser = subparsers.add_parser('super-resolution', parents=[ap], add_help=False, help='"super-resolution" help')
    sr_parser.add_argument("--in_rate",  required=True, help="input rate", type=int)
    sr_parser.add_argument("--out_rate",  required=True, help="output rate", type=int)

    dn_parser = subparsers.add_parser('denoising', parents=[ap], add_help=False, help='"denoising" help')
    dn_parser.add_argument("-r", "--rate", help="sample rate, default 16000", type=int, default=16000)

    ofdn_parser = subparsers.add_parser('overfit-dn', help='"overfit" help')
    ofsr_parser = subparsers.add_parser('overfit-sr', help='"overfit" help')

    id_parser = subparsers.add_parser('identity' , parents=[ap], add_help=False, help='"identity" help')
    id_parser.add_argument("-r", "--rate", help="sample rate, default 16000", type=int, default=16000)


    args = pa.parse_args()
    if (args.command == 'overfit-dn'):
        return overfit_dn()
    elif (args.command == 'overfit-sr'):
        return overfit_sr()
    


    variables = vars(args)
    count = variables['count']
    out = variables['out']
    epochs = variables['epochs']
    batch = variables['batch']
    window = variables['window']
    stride = variables['stride']
    depth = variables['depth']
    train_n = variables['train_n']
    test_n = variables['test_n']
    load = True if variables['load'] == 1 else 0
    continue_train = True if variables['load'] == 1 else 0
    dataset = variables['dataset']
    name = variables['name']

    if (args.command == 'denoising'):
        rate = variables['rate']
        denoising(count, out, epochs, batch, window, stride, depth, rate, train_n, test_n, load, continue_train, name, dataset)

    elif (args.command == 'super-resolution'):
        in_rate = variables['in_rate']
        out_rate = variables['out_rate']
        super_resolution(count, out, epochs, batch, window, stride, depth, in_rate, out_rate, train_n, test_n, load, continue_train, name, dataset)
    elif (args.command == 'identity'):
        rate = variables['rate']
        return identity(count, out, epochs, batch, window, stride, depth, rate, train_n, test_n, load, continue_train, name, dataset)
    else:
        print("invalid argument for the model")






def init_net(depth):
    
    net = Net(depth, verbose = 0)

    if torch.cuda.is_available():
        net.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using : " + str(device))
    print("Network initialized")
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

def load_data(year=-1, train_n=-1, test_n=-1, dataset="beethoven", preprocess='upscaling', batch_size=16, args=[]):
    
    """
    train_n : number of files used as train data
    test_n : number of files used as test data
    val_n : number of files used as val data
    """
    f = SimpleFiles("/mnt/Data/Beethoven/", 0.9)
    #f = SimpleFiles("/data/lois-data/Beethoven/", 0.9)
    if dataset == 'maestro': f = MAESTROFiles("/mnt/Data/maestro-v2.0.0", year)

 
    names_train = f.get_train(train_n)
    print("Train files : " + str(names_train))
    names_test = f.get_test(test_n)
    print("Test files : " + str(names_test))

    # names_val = f.get_validation(val_n)
    # print(names_val)
    datasets_train = [get_dataset_fn(preprocess)(n, *args) for n in names_train]
    datasets_test = [get_dataset_fn(preprocess)(n, *args) for n in names_test]
    # datasets_val = [get_dataset_fn(dataset)(n, *args) for n in names_val]
    data_train = ConcatDataset(datasets_train)
    data_test = ConcatDataset(datasets_test)
    # data_val = ConcatDataset(datasets_val)
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)
    # val_loader = DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    print("Data loaded")
    return train_loader, test_loader#, val_loader

   
def train(model, loader, epochs, count, name, loss, optim, device):

    print("Training for " + str(epochs) +  " epochs, " + str(count) + " mini-batches per epoch")
    train_step = make_train_step(model, loss, optim)
    cuda = torch.cuda.is_available()
    losses = []
    for epoch in range(epochs):
        bar = progressbar.ProgressBar(max_value=count, redirect_stdout=True)
        #bar = Bar('Training', max=count)
        for x_batch, y_batch in loader:
            #print(cuda)
            if cuda: model.cuda()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)

            plt.plot(losses)
            plt.yscale('log')
            
            #bar.next()
            bar.update(len(losses))
            #print("epoch " + str(epoch))
            if (count > 0 and len(losses) >= count*(epoch+1)): break
            if (count % 100 == 0): plt.savefig('img/'+name+'_train.png')
        #bar.finish()
        plt.savefig('img/'+name+'_train.png')
        if (epoch % 5 == 0): 
            torch.save(model, "models/" + name + ".pt")
            #torch.save(model.state_dict(), "models/" + name + ".pt")
    print("Model trained")
    plt.clf()

def test(model, loader, count, name, loss,  device):
    test_step = make_test_step(model, loss)

    # Test model

    losses = []
    outputs = []
    bar = Bar('Testing', max=count)
    with torch.no_grad():
        for x_test, y_test in loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            loss, y_test_hat = test_step(x_test, y_test)
            losses.append(loss)
    
            outputs.append(y_test_hat)
            bar.next()
            if (count > 0 and len(losses) >= count ): break
        bar.finish()
    plt.plot(losses)
    plt.yscale('log')
    plt.savefig('img/'+name+'_test.png')
    plt.clf()
    return outputs

def create_output_audio(outputs, rate, name, window, stride, batch):
    #outputs = [torch.flatten(x, 0)[None, None, :] for x in outputs]# addedd to hansdle batches in output 
    outputs = [torch.chunk(x, batch, dim=0) for x in outputs]
    outputs = [val for sublist in outputs for val in sublist]
    out = cut_and_concat_tensors(outputs, window, stride)
    out_formated = out.reshape((1, out.size()[2]))
    torchaudio.save("out/"+name+".wav", out_formated, rate, precision=16, channels_first=True)

def super_resolution(count, out, epochs, batch, window, stride, depth, in_rate, out_rate, train_n, test_n, load, continue_train, name, dataset):

    # Init net and cuda
    net, device = init_net(depth)
    # Open data, split train and val set
    train_loader, test_loader = load_data(train_n=train_n, test_n=test_n, dataset=dataset, preprocess='upscaling', batch_size=batch, args=[window, stride, in_rate, out_rate])

    adam = optim.Adam(net.parameters(), lr=0.0001)

    if load: 
        net = torch.load("models/" + name + ".pt")
        net.eval()

    else: train(model=net, loader = train_loader, epochs=epochs, count=count, name=name, loss=nn.MSELoss(), optim=adam, device=device)
    


    # val(model=net, loader=val_loader, count=500, name=name, loss=nn.MSELoss(), device=device)
    # print("Model validated")        

    outputs = test(model=net, loader=test_loader, count=out, name=name, loss=nn.MSELoss(), device=device)
    create_output_audio(outputs = outputs, rate=out_rate, name=name, window = window, stride=stride, batch=batch)
    print("Output file created")



def denoising(count, out, epochs, batch, window, stride, depth, rate, train_n, test_n, load, continue_train, name, dataset):
     # Init net and cuda
    net, device = init_net(depth)
    print("Network initialized")
    # Open data, split train and val set

    train_loader, test_loader = load_data(train_n=train_n, test_n=test_n, dataset=dataset, preprocess='denoising', batch_size=batch,args=[window, stride, rate])
    print("Data loaded")
   
    adam = optim.Adam(net.parameters(), lr=0.0001)
    if load: 
        net = torch.load("models/" + name + ".pt")
        net.eval()
    else: train(model=net, loader = train_loader, epochs=epochs, count=count, name=name, loss=nn.MSELoss(), optim=adam, device=device)
    print("Model trained")


    # val(model=net, loader=val_loader, count=500, name=name, loss=nn.MSELoss(), device=device)
    # print("Model validated")        

    outputs = test(model=net, loader=test_loader, count=out, name=name, loss=nn.MSELoss(), device=device)
    create_output_audio(outputs = outputs, rate=rate, name=name, window = window, stride=stride, batch=batch)
    print("Output file created")

def identity(count, out, epochs, batch, window, stride, depth, rate, train_n, test_n, load, continue_train, name, dataset):

    # Init net and cuda
    net, device = init_net(depth)
    print("Network initialized")
    # Open data, split train and val set

    train_loader, test_loader = load_data(train_n=train_n, test_n=test_n, dataset=dataset, preprocess='identity', batch_size=batch, args=[window, stride, rate])
    print("Data loaded")
   
    adam = optim.Adam(net.parameters(), lr=0.0001)
    if load: 
        net = torch.load("models/" + name + ".pt")
        net.eval()
    else: train(model=net, loader = train_loader, epochs=epochs, count=count, name=name, loss=nn.MSELoss(), optim=adam, device=device)
    print("Model trained")
      

    outputs = test(model=net, loader=test_loader, count=out, name=name, loss=nn.MSELoss(), device=device)
    create_output_audio(outputs = outputs, rate=rate, name=name, window = window, stride=stride, batch=batch)
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
    test_data = AudioUpScalingDataset("MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.wav", window=1024, stride=512, compressed_rate=5000, target_rate=10000, size=100)
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





