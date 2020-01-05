
# coding=utf-8
#!/home/lois/python37/bin/python


import argparse
import os
import sys
from test import make_test_step, test

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from datasets import AudioDataset
from files import MAESTROFiles, SimpleFiles
from network import (AutoEncoder, ConditionalDiscriminator, Discriminator,
                     Generator)
from torch.utils.data import ConcatDataset, DataLoader
from train import make_train_step, train
from utils import (concat_list_tensors, create_output_audio,
                   cut_and_concat_tensors, plot, str2bool)

GAN = False


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
    ap.add_argument("--load",  help="load already trained model to evaluate file given as argument [string], default=''", type=str, default="")
    ap.add_argument("--continue", help="load already trained model to continue training [bool], default=False, not implemented yet", type=str2bool, default=False)
    ap.add_argument("--dataset", help="type of the dataset[simple|type], where 'type' is a custom dataset type implemented in load_data(), default=simple", type=str, default="simple")
    ap.add_argument("--dataset_args", help="optional arguments for specific datasets, strings separated by commas", type=str)
    ap.add_argument("--data_root", help="root of the dataset [path], default=/data/lois-data/models/maestro", type=str, default="/data/lois-data/models/maestro/")
    ap.add_argument("--rate", required=True, help="Sample rate of the output file [int], mandatory", type=int)
    ap.add_argument("--preprocessing", required=True, help="Preprocessing pipeline, a string with each step of the pipeline separated by a comma, more details in readme file", type=str)
    ap.add_argument("--loss", required=False, help="Choose the loss for the generator, [L1, L2], default='L2')", type=str, default="L2")
    ap.add_argument("--gan", required=False, help="lambda for the gan loss [float], default=0 (meaning gan disabled)", type=float, default=0)
    ap.add_argument("--ae", required=False, help="lambda for the audoencoder loss [float], default=0 (meaning autoencoder disabled)", type=float, default=0)
    ap.add_argument("--collab", required=False, help="Enable the collaborative gan [bool], default=False", type=str2bool, default=False)
    ap.add_argument("--cgan", required=False, help="Enable Conditional GAN [bool], default=False", type=str2bool, default=False)
    ap.add_argument("--lr_g", required=False, help="learning rate for the generator [float], default=0.0001", type=float, default=0.0001)
    ap.add_argument("--lr_d", required=False, help="learning rate for the discriminator [float], default=0.0001", type=float, default=0.0001)
    ap.add_argument("--lr_ae", required=False, help="learning rate for the autoencoder [float], default=0.0001", type=float, default=0.0001)
    ap.add_argument("--scheduler", required=False, help="enable the scheduler [bool], default=False", type=str2bool, default=False)

    args = ap.parse_args()
    variables = vars(args)

    print(variables)

    global ROOT
    ROOT = variables['data_root']
    
    gan = variables['gan']
    ae = variables['ae']
    collab = variables['collab']
    cgan = variables['cgan']
    count = variables['count']
    out = variables['out']
    epochs = variables['epochs']
    batch = variables['batch']
    window = variables['window']
    stride = variables['stride']
    depth = variables['depth']
    train_n = variables['train_n']
    load = variables['load']
    continue_train = variables['continue']
    dataset = variables['dataset']
    dataset_args = variables['dataset_args']
    rate = variables['rate']
    loss = variables['loss']
    preprocessing = variables['preprocessing']
    name = variables['name']
    dropout = variables['dropout']
    lr_g = variables['lr_g']
    lr_d = variables['lr_d']
    lr_ae = variables['lr_ae']
    scheduler = variables['scheduler']
    
    # If we start from the start, create the necessary folders
    if (load == ""):
        os.system("rm -rf out/" + name)
        os.system("mkdir out/" + name)
        os.system("mkdir out/" + name + "/models")
        os.system("mkdir out/" + name + "/tmp")

    # Save the command in a file for future reference
    with open("out/" + name + "/command", "w") as text_file:
        text_file.write(" ".join(sys.argv))
    

    pipeline(count, out, epochs, batch, window, stride, depth, dropout, 
            lr_g, lr_d, lr_ae, rate, loss, train_n, load, continue_train, 
            name, dataset, dataset_args, preprocessing, gan, ae, collab, cgan, scheduler)

# Initialize all the networks
def init_net(depth, dropout, window, cgan):
    

    input_shape = (1 if not cgan else 2, window)
    # Create the 3 networks
    gen = Generator(depth, dropout, verbose=0)
    discr = Discriminator(depth, dropout, input_shape, verbose=0) if not cgan else ConditionalDiscriminator(depth, dropout, input_shape, verbose=0)
    ae = AutoEncoder(depth, dropout, verbose=0)
    
    # Put them on cuda if available
    if torch.cuda.is_available():
        gen.cuda()
        discr.cuda()
        ae.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using : " + str(device))
    print("Network initialized\n")

    return gen, discr, ae, device

# Load a single file into the dataloader, used for evaluation only
def load_single(name, dataset, preprocess, batch_size, window, stride, run_name):

    dataset = AudioDataset(run_name, ROOT + "/" + name, window, stride, preprocess)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return loader

def load_data(train_n, val_n, dataset, preprocess, batch_size, window, stride, dataset_args, run_name):
    
    # Load the correct file structure according to the parameter
    f = None
    if dataset == 'simple': f = SimpleFiles(ROOT, 0.9)
    elif dataset == 'maestro': f = MAESTROFiles("/mnt/Data/maestro-v2.0.0", int(dataset_args.split(',')[0]))
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
    datasets_train = [AudioDataset(run_name, n, window, stride, preprocess) for n in names_train]
    datasets_test =  [AudioDataset(run_name, n, window, stride, preprocess) for n in names_test]
    datasets_val =   [AudioDataset(run_name, n, window, stride, preprocess, size=128, start=32) for n in names_val]

    # Since those are actually lists of datasets (one for each file), concatenate them
    data_train = ConcatDataset(datasets_train)
    data_test = ConcatDataset(datasets_test)
    data_val = ConcatDataset(datasets_val)
    
    # Put the data into dataloaders with the correct batch size, and with some shuffle
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    print("Data loaded")
    return train_loader, test_loader, val_loader



def pipeline(count, out, epochs, batch, window, stride, depth, dropout, lr_g, lr_d, lr_ae, out_rate, loss, train_n, load, continue_train, name, dataset, dataset_args, preprocessing, gan_lb, ae_lb, collab, cgan, scheduler):
    # Init net and cuda
    gen, discr, ae, device = init_net(depth=depth, dropout=dropout, window=window, cgan=cgan)
    
    # Open data, split train and val set, depending on if we want to do a full training or only the evaluation
    train_loader, test_loader, val_loader = None, None, None
    if load != "": test_loader = load_single(name=load, dataset=dataset, preprocess=preprocessing, batch_size=batch, window=window, stride=stride, run_name=name)
    else: train_loader, test_loader, val_loader = load_data(train_n=train_n, val_n=1, dataset=dataset, dataset_args=dataset_args, preprocess=preprocessing, batch_size=batch, window=window, stride=stride, run_name=name)


    # Create our optimizers
    adam_gen = optim.Adam(gen.parameters(), lr=lr_g)
    adam_discr = optim.Adam(discr.parameters(), lr=lr_d)
    adam_ae = optim.Adam(ae.parameters(), lr=lr_ae)

    # If want to load a pre-trained model, load everything usefull from the .tar file
    if load != "": 
        checkpoint = torch.load("out/" + name + "/models/model.tar")

        gen.load_state_dict(checkpoint['gen_state_dict'])
        if gan_lb: discr.load_state_dict(checkpoint['discr_state_dict'])
        if ae_lb: ae.load_state_dict(checkpoint['ae_state_dict'])

        adam_gen.load_state_dict(checkpoint['optim_g_state_dict'])
        if gan_lb: adam_discr.load_state_dict(checkpoint['optim_d_state_dict'])
        if ae_lb: adam_ae.load_state_dict(checkpoint['optim_ae_state_dict'])

    # If we want to train the model, simply call the train function
    if ((load == "") or continue_train): 
        train(gen=gen, discr=discr, ae=ae, loader=train_loader, val=val_loader, epochs=epochs, count=count, name=name, loss=loss, optim_g=adam_gen, optim_d=adam_discr, optim_ae=adam_ae, device=device, gan=gan_lb, ae_lb=ae_lb, scheduler=scheduler, collab=collab, cgan=cgan)

    # Once it's trained, generate an improved audio by calling test
    outputs = test(gen=gen, discr=discr, ae=ae, loader=test_loader, count=out, name=name, loss=loss, device=device, gan_lb=gan_lb, ae_lb=ae_lb, collab=collab, cgan=cgan)

    # Put the data together into a real file
    create_output_audio(outputs = outputs, rate=out_rate, name=name, window = window, stride=stride, batch=batch)
    print("Output file created")


    

if __name__ == "__main__":
    init()

#############
# TO REMOVE #
#############

# Essayer d'overfit avec un petit bout de musique
# Ca devrait donner un bon resultat
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


# CGAN
# Overfit
# Tester ae
# Faire slides
# Quand meme mettre les loss des différents moodèles
