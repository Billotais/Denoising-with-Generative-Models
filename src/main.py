
# coding=utf-8
#!/home/lois/python37/bin/python


import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
#from progress.bar import Bar
#import progressbar
from torch.utils.data import ConcatDataset, DataLoader

from datasets import AudioUpScalingDataset, AudioWhiteNoiseDataset, AudioIDDataset, AudioDataset
from files import MAESTROFiles, SimpleFiles
from network import Generator, Discriminator, AutoEncoder, ConditionalDiscriminator
from utils import (concat_list_tensors, cut_and_concat_tensors, plot, create_output_audio)
from train import make_train_step, train
from test import  make_test_step, test
import numpy as np
from utils import str2bool

import torch.optim




GAN = False

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')  

def init():

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
    #ap.add_argument("--special", required=False, help="Use a special pipeline in the code", type=str, default="normal")
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
    #if (variables['special'] == 'overfit-sr'):
    #    return overfit_sr()
  
    print(variables)
    global ROOT
    
    gan = variables['gan']
    ae = variables['ae']
    collab = variables['collab']
    cgan = variables['cgan']
    ROOT = variables['data_root']
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
    #print(scheduler)
    
    if (load == ""):
        os.system("rm -rf out/" + name)
        os.system("mkdir out/" + name)
        os.system("mkdir out/" + name + "/models")
        os.system("mkdir out/" + name + "/tmp")

    #os.system(" ".join(sys.argv) + " > " + name + "/command")

    with open("out/" + name + "/command", "w") as text_file:
        text_file.write(" ".join(sys.argv))
    #print("".join(sys.argv))
    pipeline(count, out, epochs, batch, window, stride, depth, dropout, lr_g, lr_d, lr_ae, rate, loss, train_n, load, continue_train, name, dataset, dataset_args, preprocessing, gan, ae, collab, cgan, scheduler)


def init_net(depth, dropout, input_shape, cgan):
    
    gen = Generator(depth, dropout, verbose=0)
    discr = Discriminator(depth, dropout, input_shape, verbose=0) if not cgan else ConditionalDiscriminator(depth, dropout, input_shape, verbose=0)
    ae = AutoEncoder(depth, dropout, verbose=0)
    

    if torch.cuda.is_available():
        gen.cuda()
        discr.cuda()
        ae.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using : " + str(device))
    print("Network initialized")
    # print(gen)
    # print(discr)
    return gen, discr, ae, device


def load_single(name, dataset, preprocess, batch_size, window, stride, run_name):

    dataset = AudioDataset(run_name, ROOT + "/" + name, window, stride, preprocess)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return loader

def load_data(train_n, val_n, dataset, preprocess, batch_size, window, stride, dataset_args, run_name):
    
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
    names_val = f.get_val()
    print("\nVal files : " + str(names_val))
    names_test = f.get_test()
    print("\nTest files : " + str(names_test) + "\n")


    #names_val = f.get_validation(val_n)
    #print(names_val)


    datasets_train = [AudioDataset(run_name, n, window, stride, preprocess) for n in names_train]
    datasets_test =  [AudioDataset(run_name, n, window, stride, preprocess) for n in names_test]
    datasets_val =   [AudioDataset(run_name, n, window, stride, preprocess, 128, 32) for n in names_val]

    data_train = ConcatDataset(datasets_train)
    data_test = ConcatDataset(datasets_test)
    data_val = ConcatDataset(datasets_val)
    
    train_loader = DataLoader(dataset=data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=data_val, batch_size=batch_size, shuffle=True)

    print("Data loaded")
    return train_loader, test_loader, val_loader



def pipeline(count, out, epochs, batch, window, stride, depth, dropout, lr_g, lr_d, lr_ae, out_rate, loss, train_n, load, continue_train, name, dataset, dataset_args, preprocessing, gan_lb, ae_lb, collab, cgan, scheduler):
    # Init net and cuda
    print(cgan)
    print(1 if not cgan else 2)
    gen, discr, ae, device = init_net(depth, dropout, (1 if not cgan else 2, window), cgan)
    
    # Open data, split train and val set
    train_loader, test_loader, val_loader = None, None, None

    if load != "": test_loader = load_single(name=load, dataset=dataset, preprocess=preprocessing, batch_size=batch, window=window, stride=stride, run_name=name)
    else: train_loader, test_loader, val_loader = load_data(train_n=train_n, val_n=1, dataset=dataset, dataset_args=dataset_args, preprocess=preprocessing, batch_size=batch, window=window, stride=stride, run_name=name)

    adam_gen = optim.Adam(gen.parameters(), lr=lr_g)
    adam_discr = optim.Adam(discr.parameters(), lr=lr_d)
    adam_ae = optim.Adam(ae.parameters(), lr=lr_ae)
    loss = nn.MSELoss() if loss == "L2" else nn.L1Loss()

    if load != "": 

        checkpoint = torch.load("out/" + name + "/models/model.tar")
        gen.load_state_dict(checkpoint['gen_state_dict'])
        if gan_lb: discr.load_state_dict(checkpoint['discr_state_dict'])
        if ae_lb: ae.load_state_dict(checkpoint['ae_state_dict'])
        adam_gen.load_state_dict(checkpoint['optim_g_state_dict'])
        if gan_lb: adam_discr.load_state_dict(checkpoint['optim_d_state_dict'])
        if ae_lb: adam_ae.load_state_dict(checkpoint['optim_ae_state_dict'])
        # gen = torch.load("out/" + name + "models/model_gen.pt")
        # gen.eval()
        # discr = torch.load("out/" + name + "models/model_discr.pt")
        # discr.eval()

    if ((load == "") or continue_train): 
        train(gen=gen, discr=discr, ae=ae, loader=train_loader, val=val_loader, epochs=epochs, count=count, name=name, loss_fn=loss, optim_g=adam_gen, optim_d=adam_discr, optim_ae=adam_ae, device=device, gan=gan_lb, ae_lb=ae_lb, scheduler=scheduler, collab=collab, cgan=cgan)

    outputs = test(gen=gen, discr=discr, ae=ae, loader=test_loader, count=out, name=name, loss=loss, device=device, gan_lb=gan_lb, ae_lb=ae_lb, collab=collab, cgan=cgan)
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



