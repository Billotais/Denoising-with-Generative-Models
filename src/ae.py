# Main file for the autoencoder

from datasets import IDDataset
from files import MAESTROFiles, SimpleFiles
from network import (AutoEncoder)


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