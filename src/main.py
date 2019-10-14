import argparse


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader

from network import Net
from utils import make_train_step, make_test_step, concat_list_tensors, cut_and_concat_tensors
from datasets import AudioWhiteNoiseDataset, AudioUpScalingDataset

from progress.bar import Bar
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()

    subparsers = ap.add_subparsers(title="commands", dest="command")

    sr_parser = subparsers.add_parser('super-resolution', help='"super-resolution" help')
   
    sr_parser.add_argument("-c", "--count", required=True, help="count : number of samples used for training", type=int)
    sr_parser.add_argument("-e", "--epochs", help="epochs : number of epochs", type=int, default=1)
    sr_parser.add_argument("-b", "--batch", help="batch : size of a training batch", type=int, default=16)
    sr_parser.add_argument("-w", "--window", help="window : size of the sliding window", type=int, default=1024)
    sr_parser.add_argument("-s", "--stride", help="stride : stride of the sliding window", type=int, default=512)
    sr_parser.add_argument("-d", "--depth", help="depth : number of layers of the network", type=int, default=8)

    sr_parser.add_argument("--in_rate",  required=True, help="in_rate : input_rate", type=int)
    sr_parser.add_argument("--out_rate",  required=True, help="out_rate : output_rate", type=int)


    dn_parser = subparsers.add_parser('denoising', help='"denoising" help')
   
    dn_parser.add_argument("-c", "--count", required=True, help="count : number of samples used for training", type=int)
    dn_parser.add_argument("-e", "--epochs", help="epochs : number of epochs", type=int, default=1)
    dn_parser.add_argument("-b", "--batch", help="batch : size of a training batch", type=int, default=16)
    dn_parser.add_argument("-w", "--window", help="window : size of the sliding window", type=int, default=1024)
    dn_parser.add_argument("-s", "--stride", help="stride : stride of the sliding window", type=int, default=512)
    dn_parser.add_argument("-d", "--depth", help="depth : number of layers of the network", type=int, default=8)

    dn_parser.add_argument("-r", "--rate", help="rate : rample rate", type=int, default=10000)
    args = ap.parse_args()



    variables = vars(args)
    count = variables['count']
    epochs = variables['epochs']
    batch = variables['batch']
    window = variables['window']
    stride = variables['stride']
    depth = variables['depth']

    if (args.command == 'denoising'):
        rate = variables['rate']
        denoising(count, epochs, batch, window, stride, depth, rate)

    elif (args.command == 'super-resolution'):
        in_rate = variables['int_rate']
        out_rate = variables['out_rate']
        super_resolution(count, epochs, batch, window, stride, depth, in_rate, out_rate)

    else:
        print("invalid argument for the model")



def denoising(count, epochs, batch, window, stride, depth, rate):

    # Init network and cuda if available

    VAL_SAMPLES = 100
    net = Net(depth, verbose = 0)

    if torch.cuda.is_available():
        net.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using : " + str(device))

    # Open data, split in train and val

    filename = "/mnt/Data/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
    data = AudioWhiteNoiseDataset(filename, window=window, stride=stride, rate=rate)

    train_count = int(len(data)*count / (count + VAL_SAMPLES))
    train_data, val_data = torch.utils.data.random_split(data, [train_count, len(data) - train_count])

    train_data_sample = torch.utils.data.random_split(train_data, [count, len(train_data) - count])[0]
    val_data_sample = torch.utils.data.random_split(val_data, [VAL_SAMPLES, len(val_data) - VAL_SAMPLES])[0]

    print("Training data : " + str(len(train_data_sample)))
    print("Validation data : "  + str(len(val_data_sample)))


    # Load data into batches

    train_loader = DataLoader(dataset=train_data_sample, batch_size=batch, shuffle=True)
    train_step = make_train_step(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=0.0001))
    val_loader = DataLoader(dataset=val_data_sample, batch_size=batch, shuffle=False)
    val_step = make_test_step(net, nn.MSELoss())

    # Train model

    n_epochs = epochs
    losses = []
    for epoch in range(n_epochs):
        for x_batch, y_batch in Bar('Training', suffix='%(percent)d%%').iter(train_loader):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            loss = train_step(x_batch, y_batch)
            losses.append(loss)

            plt.plot(losses)
            plt.yscale('log')
            plt.savefig('img/loss_train_denoising.png')
            plt.show()

    # Validate model
    losses = []
    for x_val, y_val in Bar('Validation', suffix='%(percent)d%%').iter(val_loader):
        
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        loss, y_val_hat = val_step(x_val, y_val)
        losses.append(loss)

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('img/loss_val_denoising.png')
        plt.show()

    # Load test data
    filename = "/mnt/Data/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"

    test_data = AudioWhiteNoiseDataset(filename, window=window, stride=stride, rate=rate, size=500)

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    test_step = make_test_step(net, nn.MSELoss())

    # Test model
    losses = []
    outputs = []
    for x_test, y_test in Bar('Testing', suffix='%(percent)d%%').iter(test_loader):
        
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        loss, y_test_hat = val_step(x_test, y_test)
        losses.append(loss)

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('img/loss_test_denoising.png')
        plt.show()
        outputs.append(y_test_hat)

    # Save output file
    out = cut_and_concat_tensors(outputs, window, stride)
    out_formated = out.reshape((1, out.size()[2]))
    torchaudio.save("out/denoising-out.wav", out_formated, rate, precision=16, channels_first=True)


def super_resolution(count, epochs, batch, window, stride, depth, in_rate, out_rate):

    # Init net and cuda

    VAL_SAMPLES = 100
    net = Net(depth, verbose = 0)

    if torch.cuda.is_available():
        net.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using : " + str(device))

    # Open data, split train and val set

    filename = "/mnt/Data/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"
    data = AudioUpScalingDataset(filename, window=window, stride=stride, compressed_rate=in_rate, target_rate=out_rate)

    train_count = int(len(data)*count / (count + VAL_SAMPLES))
    train_data, val_data = torch.utils.data.random_split(data, [train_count, len(data) - train_count])

    train_data_sample = torch.utils.data.random_split(train_data, [count, len(train_data) - count])[0]
    val_data_sample = torch.utils.data.random_split(val_data, [VAL_SAMPLES, len(val_data) - VAL_SAMPLES])[0]

    print("Training data : " + str(len(train_data_sample)))
    print("Validation data : "  + str(len(val_data_sample)))

    # Load data into batches

    train_loader = DataLoader(dataset=train_data_sample, batch_size=batch, shuffle=True)
    train_step = make_train_step(net, nn.MSELoss(), optim.Adam(net.parameters(), lr=0.0001))
    val_loader = DataLoader(dataset=val_data_sample, batch_size=batch, shuffle=False)
    val_step = make_test_step(net, nn.MSELoss())

    # Train model
    n_epochs = epochs
    losses = []
    for epoch in range(n_epochs):
        for x_batch, y_batch in Bar('Training', suffix='%(percent)d%%').iter(train_loader):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = train_step(x_batch, y_batch)
            losses.append(loss)

            plt.plot(losses)
            plt.yscale('log')
            plt.savefig('img/loss_train.png')
            plt.show()
            

    # Validation

    losses = []
    for x_val, y_val in Bar('Validation', suffix='%(percent)d%%').iter(val_loader):
        x_val = x_val.to(device)
        y_val = y_val.to(device)

        loss, y_val_hat = val_step(x_val, y_val)
        losses.append(loss)

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('loss_val.png')
        plt.show()


    # Load test data
    filename = "/mnt/Data/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.wav"

    test_data = AudioUpScalingDataset(filename, window=window, stride=stride, compressed_rate=in_rate, target_rate=out_rate, size=1000)

    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)
    test_step = make_test_step(net, nn.MSELoss())

    # Test model

    losses = []
    outputs = []
    for x_test, y_test in Bar('Testing', suffix='%(percent)d%%').iter(test_loader):
        x_test = x_test.to(device)
        y_test = y_test.to(device)

        loss, y_test_hat = val_step(x_test, y_test)
        losses.append(loss)

        plt.plot(losses)
        plt.yscale('log')
        plt.savefig('loss_test.png')
        plt.show()
        outputs.append(y_test_hat)



    # Save output file
    out = cut_and_concat_tensors(outputs, window, stride)
    out_formated = out.reshape((1, out.size()[2]))
    torchaudio.save("out/super-resolution-out.wav", out_formated, out_rate, precision=16, channels_first=True)



if __name__ == "__main__":
    main()



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
