
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
from progress.bar import Bar
import progressbar
from torch.utils.data import ConcatDataset, DataLoader

from datasets import AudioUpScalingDataset, AudioWhiteNoiseDataset, AudioIDDataset, AudioDataset
from files import MAESTROFiles, SimpleFiles
from network import Generator, Discriminator
from utils import (concat_list_tensors, cut_and_concat_tensors, plot, create_output_audio)
from train import make_train_step_gan
from test import  make_test_step_gan
import numpy as np

import torch.optim




GAN = False

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')  

def init():

    ap = argparse.ArgumentParser()
   
    ap.add_argument("-c", "--count", required=False, help="number of mini-batches used for training", type=int, default=50)
    ap.add_argument("-o", "--out", required=False, help="number of samples to output", type=int, default=500)
    ap.add_argument("-e", "--epochs", help="number of epochs, default 1", type=int, default=1)
    ap.add_argument("-b", "--batch", help="size of a training batch, default 32", type=int, default=32)
    ap.add_argument("-w", "--window", help="size of the sliding window, default 2048", type=int, default=2048)
    ap.add_argument("-s", "--stride", help="stride of the sliding window, default 1024", type=int, default=1024)
    ap.add_argument("-d", "--depth", help="number of layers of the network, default 4", type=int, default=4)
    ap.add_argument("-n", "--name",  required=True, help="name of model", type=str)
    ap.add_argument("--dropout", help="value fo the dropout in the entwork", type=float, default=0.5)
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
    ap.add_argument("--gan", required=False, help="lambda for the gan", type=float, default=0)
    ap.add_argument("--lr_g", required=False, help="learning rate for the generator", type=float, default=0.0001)
    ap.add_argument("--lr_d", required=False, help="learning rate for the discriminator", type=float, default=0.0001)

    args = ap.parse_args()
    variables = vars(args)
    if (variables['special'] == 'overfit-sr'):
        return overfit_sr()
  
    global ROOT
    global GAN
    GAN = variables['gan']
    ROOT = variables['data_root']
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
    
    rate = variables['rate']
    preprocessing = variables['preprocessing']
    name = variables['name']
    dropout = variables['dropout']
    lr_g = variables['lr_g']
    lr_d = variables['lr_d']
    

    os.system("rm -rf out/" + name)
    os.system("mkdir out/" + name)
    os.system("mkdir out/" + name + "/models")
    os.system("mkdir out/" + name + "/tmp")

    #os.system(" ".join(sys.argv) + " > " + name + "/command")

    with open("out/" + name + "/command", "w") as text_file:
        text_file.write(" ".join(sys.argv))
    #print("".join(sys.argv))
    pipeline(count, out, epochs, batch, window, stride, depth, dropout, lr_g, lr_d, rate, train_n, test_n, load, continue_train, name, dataset, dataset_args, preprocessing)
    






def init_net(depth, dropout, input_shape):
    
    gen = Generator(depth, dropout, verbose = 0)
    discr = Discriminator(depth, dropout, input_shape, verbose = 0)

    if torch.cuda.is_available():
        gen.cuda()
        discr.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using : " + str(device))
    print("Network initialized")
    print(gen)
    print(discr)
    return gen, discr, device

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

   
def train(gen, discr, loader, val, epochs, count, name, loss, optim_g, optim_d, device):


    print("Training for " + str(epochs) +  " epochs, " + str(count) + " mini-batches per epoch")
    
    train_step = make_train_step_gan(gen, discr, loss, 0.2, optim_g, optim_d, GAN)
    test_step = make_test_step_gan(gen, discr, loss, GAN)

    cuda = torch.cuda.is_available()
    losses = []
    val_losses = []
    losses_gan = []
    val_losses_gan = []
    loss_buffer = []
    loss_buffer_gan = []

    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_g, verbose=True)
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_d, verbose=True)
   

    for epoch in range(1, epochs+1):

        

        # correct the count variable if needed
        total = len(loader)
        if (total < count or count < 0): 
            count = total 
        
        bar = progressbar.ProgressBar(max_value=count)
        curr_count = 0

        for x_batch, y_batch in loader:
            bar.update(curr_count)
            if cuda: 
                gen.cuda()
                discr.cuda()

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Train using the current batch
            loss, loss_gan = train_step(x_batch, y_batch)
            loss_buffer.append(loss)
            loss_buffer_gan.append(loss_gan)
            # Stop if count reached
            curr_count += 1
            if (curr_count >= count): break

            # If 100 batches done
            if (len(loss_buffer) % 100 == 0):
                #print("c100 bactches")
                # Get average train loss
                losses.append(sum(loss_buffer)/len(loss_buffer))
                losses_gan.append(sum(loss_buffer_gan)/len(loss_buffer_gan))
                loss_buffer = []
                loss_buffer_gan = []

                # Compute average test loss
                val_loss_buffer = []
                val_loss_buffer_gan = []
                for x_val, y_val in val:
                    if cuda: 
                        gen.cuda()
                        discr.cuda()
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)

                    loss, loss_gan, _ = test_step(x_val, y_val)
                    val_loss_buffer.append(loss)
                    val_loss_buffer_gan.append(loss_gan)
                val_losses.append(sum(val_loss_buffer)/len(val_loss_buffer))
                val_losses_gan.append(sum(val_loss_buffer_gan)/len(val_loss_buffer_gan))
                # Every 500, plot and decrease lr in needed
                if (len(losses) % 5 == 0):
                    plot(losses, val_losses, losses_gan, val_losses_gan, name, GAN)
                    scheduler_g.step(val_losses[-1])
                    scheduler_d.step(val_losses_gan[-1])

        
                    

        # Save the model for the epoch
        torch.save(gen, "out/" + name + "/models/model_gen_" + str(epoch) + ".pt")
        torch.save(discr, "out/" + name + "/models/model_discr_" + str(epoch) + ".pt")
        np.save('out/' + name + '/loss_train.npy', np.array(losses))
        np.save('out/' + name + '/loss_test.npy', np.array(val_losses))
        np.save('out/' + name + '/loss_train_gan.npy', np.array(losses_gan))
        np.save('out/' + name + '/loss_test_gan.npy', np.array(val_losses_gan))

       

        

        
    print("Model trained")
    plot(losses, val_losses, losses_gan, val_losses_gan, name, GAN)

def test(gen, discr, loader, count, name, loss,  device):
    test_step = make_test_step_gan(gen, discr, loss, GAN)

    # Test model

    cuda = torch.cuda.is_available()
    losses = []
    outputs = []
    bar = Bar('Testing', max=count)
    with torch.no_grad():
        for x_test, y_test in loader:
            if cuda: 
                gen.cuda()
                discr.cuda()
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            loss, _, y_test_hat = test_step(x_test, y_test)
            losses.append(loss)
    
            outputs.append(y_test_hat.to('cpu'))
            bar.next()
            if (count > 0 and len(losses) >= count ): break
        bar.finish()
    plt.plot(losses)
    plt.yscale('log')
    plt.savefig('out/'+name+'/test.png')
    plt.clf()
    return outputs



def pipeline(count, out, epochs, batch, window, stride, depth, dropout, lr_g, lr_d, out_rate, train_n, test_n, load, continue_train, name, dataset, dataset_args, preprocessing):
    # Init net and cuda
    gen, discr, device = init_net(depth, dropout, (1, window))
    # Open data, split train and val set
    train_loader, test_loader, val_loader = load_data(train_n=train_n, test_n=test_n, val_n=1, dataset=dataset, dataset_args=dataset_args, preprocess=preprocessing, batch_size=batch, window=window, stride=stride, run_name=name)

    adam_gen = optim.Adam(gen.parameters(), lr=lr_g)
    adam_disrc = optim.Adam(discr.parameters(), lr=lr_d)
    loss = nn.MSELoss()

    if load: 
        gen = torch.load("out/" + name + "models/model_gen.pt")
        gen.eval()
        discr = torch.load("out/" + name + "models/model_discr.pt")
        discr.eval()

    else: train(gen=gen, discr=discr, loader=train_loader, val=val_loader, epochs=epochs, count=count, name=name, loss=loss, optim_g=adam_gen, optim_d=adam_disrc, device=device)

    outputs = test(gen=gen, discr=discr, loader=test_loader, count=out, name=name, loss=nn.MSELoss(), device=device)
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






