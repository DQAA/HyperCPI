import torch
import numpy as np
import random
import time
import datetime

from tqdm import tqdm
from model_kinase import *
import timeit
from data_prepocess import Mol2HyperGraph
from rdkit import Chem
from torch.utils.tensorboard import SummaryWriter


def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def load_dataset(DATASET):
    dir_input = ('dataset/' + DATASET + '/word2vec_30/')
    smiles = np.load(dir_input + 'smiles.npy', allow_pickle=True)
    compounds = load_tensor(dir_input + 'compounds', torch.FloatTensor)
    proteins = np.load(dir_input + 'proteins.npy', allow_pickle=True)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

    graphs = []
    for smi in tqdm(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            exit(0)
        else:
            H = Mol2HyperGraph(mol)
            graphs.append(H)

    return compounds, graphs, proteins, interactions

if __name__ == "__main__":
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    DATASET = "kinase_BindingDB"
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('The code uses GPU...')
    else:
        device = 'cpu'
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    # compounds_train, adjacencies_train, proteins_train, interactions_train = load_dataset('kinase/train')
    # compounds_test, adjacencies_test, proteins_test, interactions_test = load_dataset('GPCR/test')

    # """Create a dataset and split it into train/dev/test."""
    # dataset_train = list(zip(compounds_train, adjacencies_train, proteins_train, interactions_train))
    # dataset_train = shuffle_dataset(dataset_train, 1234)
    # dataset_test = list(zip(compounds_test, adjacencies_test, proteins_test, interactions_test))
    # dataset_test = shuffle_dataset(dataset_test, 1234)
    # dataset_dev, dataset_test = split_dataset(dataset_test, 0.5)

    """Load preprocessed data."""
    compounds_train, adjacencies_train, proteins_train, interactions_train = load_dataset('kinase/train')
    compounds_dev, adjacencies_dev, proteins_dev, interactions_dev = load_dataset('BindingDB/dev')
    compounds_test, adjacencies_test, proteins_test, interactions_test = load_dataset('BindingDB/test')

    """Create a dataset and split it into train/dev/test."""
    dataset_train = list(zip(compounds_train, adjacencies_train, proteins_train, interactions_train))
    dataset_train = shuffle_dataset(dataset_train, 1234)
    dataset_dev = list(zip(compounds_dev, adjacencies_dev, proteins_dev, interactions_dev))
    dataset_test = list(zip(compounds_test, adjacencies_test, proteins_test, interactions_test))

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 42
    hid_dim = 256
    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.1
    batch = 32
    lr = 1e-4
    weight_decay = 1e-3
    decay_interval = 5
    lr_decay = 1.0
    iteration = 100
    kernel_size = 7

    encoder = Encoder(protein_dim, hid_dim, n_layers, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Predictor(encoder, decoder, device)
    # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model, batch)

    """Output files."""
    file_AUCs = 'output/result/' + DATASET + '.txt'
    file_model = 'output/model/' + DATASET
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev\tacc_dev\tF1_dev\tP_dev\tR_dev\tAUC_test\tPRC_test\tacc_test\tF1_test\tP_test\tR_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')
        
    current_datetime = datetime.datetime.now()
    log_file_name = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter('logs/' + log_file_name)

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    max_AUC_dev = 0
    total_step = 0
    best_epoch = 0
    for epoch in range(1, iteration+1):
        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, step = trainer.train(dataset_train, device, writer, total_step)
        total_step = step
        AUC_dev, PRC_dev, acc_dev, F1_dev, P_dev, R_dev = tester.test(dataset_dev)
        AUC_test, PRC_test, acc_test, F1_test, P_test, R_test = tester.test(dataset_test)
        writer.add_scalars('train', {'total loss': loss_train}, epoch)
        writer.add_scalars('valid', {'AUC_dev': AUC_dev}, epoch)
        writer.add_scalars('valid', {'PRC_dev': PRC_dev}, epoch)
        writer.add_scalars('valid', {'acc_dev': acc_dev}, epoch)
        writer.add_scalars('valid', {'F1_dev': F1_dev}, epoch)
        writer.add_scalars('valid', {'P_dev': P_dev}, epoch)
        writer.add_scalars('valid', {'R_dev': R_dev}, epoch)
        writer.add_scalars('test', {'AUC_test': AUC_test}, epoch)
        writer.add_scalars('test', {'PRC_test': PRC_test}, epoch)
        writer.add_scalars('test', {'acc_test': acc_test}, epoch)
        writer.add_scalars('test', {'F1_test': F1_test}, epoch)
        writer.add_scalars('test', {'P_test': P_test}, epoch)
        writer.add_scalars('test', {'R_test': R_test}, epoch)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev, PRC_dev, acc_dev, F1_dev, P_dev, R_dev, AUC_test, PRC_test, acc_test, F1_test, P_test, R_test]
        tester.save_AUCs(AUCs, file_AUCs)
        if AUC_dev > max_AUC_dev:
            tester.save_model(model, file_model)
            max_AUC_dev = AUC_dev
            best_epoch = epoch
        print('\t'.join(map(str, AUCs)))
    writer.close()
    print("result max AUC dev:{}, best epoch:{}".format(max_AUC_dev, best_epoch))

