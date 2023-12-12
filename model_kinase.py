import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from Radam import *
from lookahead import Lookahead
from hypergraph_attention import HAN
from word2vec import get_protein_embedding, seq_to_kmers
from gensim.models import Word2Vec


class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        x = torch.matmul(attention, V)

        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]

        return x


class Encoder(nn.Module):
    """protein feature extraction."""
    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout, device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        #self.pos_embedding = nn.Embedding(1000, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        #pos = torch.arange(0, protein.shape[1]).unsqueeze(0).repeat(protein.shape[0], 1).to(self.device)
        #protein = protein + self.pos_embedding(pos)
        #protein = [batch size, protein len,protein_dim]
        conv_input = self.fc(protein)
        # conv_input=[batch size,protein len,hid dim]
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1)
        #conv_input = [batch size, hid dim, protein len]
        for i, conv in enumerate(self.convs):
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))
            #conved = [batch size, 2*hid dim, protein len]

            #pass through GLU activation function
            conved = F.glu(conved, dim=1)
            #conved = [batch size, hid dim, protein len]

            #apply residual connection / high way
            conved = (conved + conv_input) * self.scale
            #conved = [batch size, hid dim, protein len]

            #set conv_input to conved for next loop iteration
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        # conved = [batch size,protein len,hid dim]
        conved = self.ln(conved)
        return conved



class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.ft = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Linear(hid_dim, 256)
        self.fc_2 = nn.Linear(256, 2)
        self.gn = nn.GroupNorm(8, 256)

    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)

        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src,trg_mask,src_mask)

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        # norm = torch.norm(trg, dim=2)
        # norm = F.softmax(norm, dim=1)
        # norm_expanded = norm.unsqueeze(2)
        # trg_weighted = trg * norm_expanded
        # sum = torch.sum(trg_weighted, dim=1)

        att_score = self.att(trg)
        att_score = F.softmax(att_score, dim=1)
        trg = att_score * trg
        sum = torch.sum(trg, dim=1)
        
        label = F.relu(self.fc_1(sum))
        label = self.fc_2(label)
        return label

class Predictor(nn.Module):
    def __init__(self, encoder, decoder, device, atom_dim=34):
        super().__init__()

        self.hgnn1 = HAN(in_channels=atom_dim, out_channels=256, use_attention=True)
        self.hgnn2 = HAN(in_channels=256, out_channels=256, use_attention=True)
        # self.hgnn3 = HAN(in_channels=128, out_channels=256)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.weight = nn.Parameter(torch.FloatTensor(atom_dim, 256))
        self.init_weight()
        

    def init_weight(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)  # batch size
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask


    def forward(self, compound, adj, protein, atom_num, protein_num, node_num, edge_num, size, atoms_len, edge_attr):

        compound_max_len = max(atom_num)
        protein_max_len = protein.shape[1]
        compound_mask, protein_mask = self.make_masks(atom_num, protein_num, compound_max_len, protein_max_len)

        compound = self.hgnn1(compound, adj)
        # compound = F.relu(compound)
        # compound = self.hgnn2(compound, adj)
        # compound = self.hgnn2(compound, adj)
        # compound = self.hgnn3(compound, adj)
        
        hidden_lst = []
        for i in range(len(node_num) - 1):
            start = node_num[i]
            end = node_num[i + 1]
            cur_hidden = compound[start:end, :]
            hidden_lst.append(torch.nn.functional.pad(cur_hidden, (0, 0, 0, atoms_len-cur_hidden.shape[0]), 'constant', 0))
        compound = torch.stack(hidden_lst)

        enc_src = self.encoder(protein)

        out = self.decoder(compound, enc_src, compound_mask, protein_mask)

        return out

    def __call__(self, data, train=True):

        compound, adj, protein, correct_interaction ,atom_num, protein_num, node_num, edge_num, size, atoms_len, edge_attr = data
        # compound = torch.squeeze(compound, 0)
        # adj = torch.squeeze(adj, 0)
        # atom_num = torch.squeeze(atom_num, 0)
        # protein_num = torch.squeeze(protein_num, 0)
        # compound = compound.to(self.device)
        # adj = adj.to(self.device)
        # protein = protein.to(self.device)
        # correct_interaction = correct_interaction.to(self.device)
        Loss = nn.CrossEntropyLoss()

        if train:
            predicted_interaction = self.forward(compound, adj, protein, atom_num, protein_num, node_num, edge_num, size, atoms_len, edge_attr)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss

        else:
            #compound = compound.unsqueeze(0)
            #adj = adj.unsqueeze(0)
            #protein = protein.unsqueeze(0)
            #correct_interaction = correct_interaction.unsqueeze(0)
            predicted_interaction = self.forward(compound, adj, protein, atom_num, protein_num, node_num, edge_num, size, atoms_len, edge_attr)
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores


def pack(atoms, adjs, proteins, labels, device, model):
    protein_embeddings = []
    for protein in proteins:
        protein_embeddings.append(get_protein_embedding(model, seq_to_kmers(protein)))
    proteins = protein_embeddings
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num = []
    for atom in atoms:
        atom_num.append(atom.shape[0])
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]
    protein_num = []
    for protein in proteins:
        protein_num.append(protein.shape[0])
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]
    adj_len = 0
    node_num = [0]
    edge_num = [0]
    size = [0]
    for adj in adjs:
        # adj_len = max(adj.shape[1], adj_len)
        if adj[0].shape[0] == 0:
            adj = torch.zeros((2, 2), device=device)
        adj_len += adj.shape[1]
        node_num.append(node_num[-1]+adj[0].max()+1)
        edge_num.append(edge_num[-1]+adj[1].max()+1)
        size.append(size[-1]+adj.shape[1])
    atoms_new = torch.zeros((int(node_num[-1].item()),34), device=device)
    i = 0
    for atom in atoms:
        atoms_new[node_num[i]:node_num[i+1], :] = atom
        i += 1
    adjs_new = torch.zeros((2, adj_len), device=device, dtype=torch.long)
    i = 0
    for adj in adjs:
        adjs_new[:2, size[i]:size[i+1]][0] = adj[0] + node_num[i]
        adjs_new[:2, size[i]:size[i+1]][1] = adj[1] + edge_num[i]
        i += 1
    edge_attr = None
    # edge_attr = torch.zeros((max(adjs_new[1]+1), 34), device=device)
    # for adj in adjs:
    #     edge_attr[adj[1], :] += atoms_new[adj[0]]
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        proteins_new[i, :a_len, :] = torch.Tensor(protein).to('cuda:0')
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1
    return (atoms_new, adjs_new, proteins_new, labels_new, atom_num, protein_num, node_num, edge_num, size, atoms_len, edge_attr)


class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        # w - L2 regularization ; b - not L2 regularization
        weight_p, bias_p = [], []

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        # self.optimizer = optim.Adam([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer_inner = RAdam(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer_inner, k=5, alpha=0.5).optimizer
        self.batch = batch

    def train(self, dataset, device, writer, total_step):
        model = Word2Vec.load("word2vec_30.model")
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        adjs, atoms, proteins, labels = [], [], [], []
        losses = []
        for data in tqdm(dataset):
            i = i+1
            atom, adj, protein, label = data
            adjs.append(adj)
            atoms.append(atom)
            proteins.append(protein)
            labels.append(label)
            if i % 32 == 0 or i == N:
                data_pack = pack(atoms, adjs, proteins, labels, device, model)
                loss = self.model(data_pack)
                # loss = loss / self.batch
                loss.backward()
                losses.append(loss.item())
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                atoms, adjs, proteins, labels = [], [], [], []
            else:
                continue
            if i % self.batch == 0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_step += 1
                writer.add_scalars('train', {'loss': np.mean(losses)}, total_step)
                losses = []
            loss_total += loss.item()
        return loss_total, total_step


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        model = Word2Vec.load("word2vec_30.model")
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        compounds, adjacencies, proteins, labels = [], [], [], []
        with torch.no_grad():
            for i, data in enumerate(tqdm(dataset)):
                compound, adjacencie, protein, label = data
                compounds.append(compound)
                adjacencies.append(adjacencie)
                proteins.append(protein)
                labels.append(label)
                if i % 32 == 0 or i == N:
                    data = pack(compounds, adjacencies, proteins, labels, self.model.device, model)
                    correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                    T.extend(correct_labels)
                    Y.extend(predicted_labels)
                    S.extend(predicted_scores)
                    compounds, adjacencies, proteins, labels = [], [], [], []
                else:
                    continue
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        acc = accuracy_score(T, Y)
        F1 = f1_score(T, Y)
        P = precision_score(T, Y)
        R = recall_score(T, Y)
        return AUC, PRC, acc, F1, P, R

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
