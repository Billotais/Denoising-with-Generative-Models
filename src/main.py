
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

from datasets import AudioUpScalingDataset, AudioWhiteNoiseDataset, AudioIDDataset, AudioDataset
from files import MAESTROFiles, SimpleFiles
from network import Net
from utils import (concat_list_tensors, cut_and_concat_tensors, make_test_step,
                   make_train_step)

ROOT = "/mnt/Data/maestro-v2.0.0/"
#ROOT = "/mnt/Data/Beethoven/"
ROOT = "/data/lois-data/Beethoven/"
ROOT = "/data/lois-data/models/maestro/"

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')  

def init():

    #os.system('mkdir /tmp/vita')

    ap = argparse.ArgumentParser()

   
    ap.add_argument("-c", "--count", required=False, help="number of mini-batches used for training", type=int, default=-1)
    ap.add_argument("-o", "--out", required=False, help="number of samples to output", type=int, default=500)
    ap.add_argument("-e", "--epochs", help="number of epochs, default 1", type=int, default=1)
    ap.add_argument("-b", "--batch", help="size of a training batch, default 32", type=int, default=32)
    ap.add_argument("-w", "--window", help="size of the sliding window, default 2048", type=int, default=2048)
    ap.add_argument("-s", "--stride", help="stride of the sliding window, default 1024", type=int, default=1024)
    ap.add_argument("-d", "--depth", help="number of layers of the network, default 4", type=int, default=4)
    ap.add_argument("-n", "--name",  required=True, help="name of model", type=str)
    ap.add_argument("--train_n", help="number of songs used to train, default 1", type=int, default=1)
    ap.add_argument("--test_n", help="number of songs used to test, default 1", type=int, default=1)
    ap.add_argument("--load",  help="load already trained model to evaluate ? 0 or 1, default 0", type=int, default=0)
    ap.add_argument("--continue", help="load already trained model to continue training ? 0 or 1, default 0", type=int, default=0)
    ap.add_argument("--dataset", help="type of the dataset : 'simple' or 'name of dataset type'", type=str, default="simple")
    ap.add_argument("--dataset_args", help="optional arguments for specific datasets, string separated by commas", type=str)
    ap.add_argument("--data_root", help="root of the dataset", type=str, default="/data/lois-data/models/maestro/")
    ap.add_argument("--rate", required=True, help="Sample rate of the output file", type=int)
    ap.add_argument("--preprocessing", required=True, help="Preprocessing pipeline, a string with each step of the pipeline separated by a comma", type=str)
    ap.add_argument("--special", required=False, help="Use a special pipeline in the code", type=str, default="normal")
   



    print("salut")
    args = ap.parse_args()
    variables = vars(args)

    print(variables)
    if (variables['special'] == 'overfit-sr'):
        return overfit_sr()
    


    
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
    continue_train = True if variables['continue'] == 1 else 0
    dataset = variables['dataset']
    dataset_args = variables['dataset_args']
    global ROOT
    ROOT = variables['data_root']
    rate = variables['rate']
    preprocessing = variables['preprocessing']
    name = variables['name']
    

    pipeline(count, out, epochs, batch, window, stride, depth, rate, train_n, test_n, load, continue_train, name, dataset, dataset_args, preprocessing)
    






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

def load_data(train_n, test_n, val_n, dataset, preprocess, batch_size, window, stride, dataset_args, run_name):
    
    """
    train_n : number of files used as train data
    test_n : number of files used as test data
    val_n : number of files used as val data
    """
    f = None
    if dataset == 'simple':
        f = SimpleFiles(ROOT, 0.9)
    elif dataset == 'maestro': f = MAESTROFiles("/mnt/Data/maestro-v2.0.0", int(dataset_args.split(',')[0]))
    else: 
        print("unknow dataset type")
        exit()

 
    names_train = f.get_train(train_n)
    print("Train files : " + str(names_train))
    names_test = f.get_test(test_n)
    print("Test files : " + str(names_test))

    #names_val = f.get_validation(val_n)
    #print(names_val)


    datasets_train = [AudioDataset(run_name, n, window, stride, preprocess) for n in names_train]
    datasets_test =  [AudioDataset(run_name, n, window, stride, preprocess) for n in names_test]
    #datasets_val = [get_dataset_fn(dataset)(n, *args) for n in names_val]
    data_train = ConcatDataset(datasets_train)
    data_test = ConcatDataset(datasets_test)
    data_train, data_val = torch.utils.data.random_split(data_train, [len(data_train)-10, 10])
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    print("Data loaded")
    return train_loader, test_loader, val_loader

   
def train(model, loader, val, epochs, count, name, loss, optim, device):

    print("Training for " + str(epochs) +  " epochs, " + str(count) + " mini-batches per epoch")
    train_step = make_train_step(model, loss, optim)
    test_step = make_test_step(model, loss)

    cuda = torch.cuda.is_available()
    losses = []
    val_avg_loss = []
    for epoch in range(epochs):

        # correct the count variable if needed
        total = len(loader)
        if (total < count or count < 0): 
            count = total 
        
        bar = progressbar.ProgressBar(max_value=count)
        curr_count = 0
        temp_losses = []
        for x_batch, y_batch in loader:
            bar.update(curr_count)
            if cuda: model.cuda()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Train using the current bathc
            loss = train_step(x_batch, y_batch)
            temp_losses.append(loss)
            # Stop if count reached
            curr_count += 1
            if (curr_count >= count): break
            # Update image every 100 mini-batches
            if (curr_count % 100 == 0):
                losses.append(sum(temp_losses)/100)
                temp_losses = []
                plt.plot(losses)
                plt.yscale('log')
                plt.savefig('img/'+name+'_train.png')
                plt.clf()
            
        # Update image final time
        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('img/'+name+'_train.png') 
        plt.clf()       

        # Save the model for the epoch
        torch.save(model, "models/" + name + "-" + str(epoch) + ".pt")

        # Validate model for this epoch
        val_losses = []
        for x_val, y_val in val:
            if cuda: model.cuda()
            x_val = x_val.to(device)
            y_val = y_val.to(device)

            loss, _ = test_step(x_val, y_val)
            val_losses.append(loss)

        # Plot the validation graph
        val_avg_loss.append(sum(val_losses)/len(val_losses))
        plt.plot(val_avg_loss)
        plt.yscale('log')
        plt.savefig('img/'+name+'_val.png') 
        plt.clf() 

        
    print("Model trained")
    plt.clf()

def test(model, loader, count, name, loss,  device):
    test_step = make_test_step(model, loss)

    # Test model

    cuda = torch.cuda.is_available()
    losses = []
    outputs = []
    bar = Bar('Testing', max=count)
    with torch.no_grad():
        for x_test, y_test in loader:
            if cuda: model.cuda()
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            loss, y_test_hat = test_step(x_test, y_test)
            losses.append(loss)
    
            outputs.append(y_test_hat.to('cpu'))
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

def pipeline(count, out, epochs, batch, window, stride, depth, out_rate, train_n, test_n, load, continue_train, name, dataset, dataset_args, preprocessing):
    # Init net and cuda
    net, device = init_net(depth)
    # Open data, split train and val set
    train_loader, test_loader, val_loader = load_data(train_n=train_n, test_n=test_n, val_n=1, dataset=dataset, dataset_args=dataset_args, preprocess=preprocessing, batch_size=batch, window=window, stride=stride, run_name=name)

    adam = optim.Adam(net.parameters(), lr=0.0001)
    loss = nn.MSELoss()

    if load: 
        net = torch.load("models/" + name + ".pt")
        net.eval()

    else: train(model=net, loader=train_loader, val=val_loader, epochs=epochs, count=count, name=name, loss=loss, optim=adam, device=device)

    outputs = test(model=net, loader=test_loader, count=out, name=name, loss=nn.MSELoss(), device=device)
    create_output_audio(outputs = outputs, rate=out_rate, name=name, window = window, stride=stride, batch=batch)
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
    test_data = AudioUpScalingDataset("/mnt/Data/maestro-v2.0.0/2004/MIDI-Unprocessed_XP_22_R2_2004_01_ORIG_MID--AUDIO_22_R2_2004_04_Track04_wav.wav", window=1024, stride=512, compressed_rate=5000, target_rate=10000, size=100)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    test_step = make_test_step(net, nn.MSELoss())
    # Train model
    n_epochs = 5000
    losses = []
    for epoch in range(n_epochs):
        for x_batch, y_batch in Bar('Training - epoch ' + str(epoch), suffix='%(percent)d%%').iter(train_loader):

            x_batch = x_batch.to(device)*1000
            y_batch = y_batch.to(device)*1000
            
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

###################################################3

# Regarder si je peux run le code du paper
# Essayer de run le maestro dataset (quelques musiques, chanson calme).

# essayer de scale *1000 pour l'overfit pour eviter les artefacts à la fin sur maestro
# training normal maestro
# Augmentation : noise + reverberation. Pouvoir chosiri avec arguments (quaniité, un des deux ou les deux, ...)
# -> avoir un module de preprocessing, ou on choisit quelle combinaison de bruit, downsampling, reverb on veut.
# reverb cpu en priorité

# regarder code du paper pour le piano pour essayer de trouver quels fichier utiliser pour loverfit

###################################################

# Verifier que la reverb sox est bien alignée avec le fichier orginal

# Run le code du paper sur mon propre split
# Run mon code avec les meme inputs que eux

# faire les slides
    # click click
    # deux dataset
    # ajouter gan ?
    # essayé de refaire le paper
    # reverb / noise / super-res
    # schéma du network, skip connection
    #  






