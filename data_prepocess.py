import numpy as np
import pandas as pd
from rdkit import Chem
import torch
from torch.utils.data import Dataset
import dgl
from dgl.dataloading import GraphDataLoader
import dgl.sparse as dglsp
from rdkit.Chem.BRICS import FindBRICSBonds
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*')  
import os


fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def GetFragment(mol):
    break_bonds = [mol.GetBondBetweenAtoms(i[0][0],i[0][1]).GetIdx() for i in FindBRICSBonds(mol)]
    if break_bonds == []:
        tmp = mol
    else:
        tmp = Chem.FragmentOnBonds(mol,break_bonds,addDummies=False)
    frags_idx_lst = Chem.GetMolFrags(tmp)
    result_ap = {}
    pharm_id = 0
    for frag_idx in frags_idx_lst:
        for atom_id in frag_idx:
            result_ap[atom_id] = pharm_id
        pharm_id += 1
    return result_ap

def Mol2HyperGraph(mol):
    
    # build graphs
    edge_types = [('a','b','a'),('a','j','p')]

    edges = {k:[] for k in edge_types}
    # if mol.GetNumAtoms() == 1:
    #     g = dgl.heterograph(edges, num_nodes_dict={'a':1,'p':1})
    # else:
    result_ap = GetFragment(mol)

    for bond in mol.GetBonds(): 
        edges[('a','b','a')].append([bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()])

    src = []
    tgt = []
    for edge in edges['a','b','a']:
        src.append(edge[0])
        tgt.append(edge[1])
    g2 = dgl.graph((src, tgt))
    g2 = dgl.to_bidirected(g2)

    for k,v in result_ap.items():
        edges[('a','j','p')].append([k,v])
    
    H1, H2 = [], []
    tot = 0
    for edge in edges[('a', 'b', 'a')]:
        H1.append(edge[0])
        H2.append(tot)
        H1.append(edge[1])
        H2.append(tot)
        tot += 1
    mp = {}
    for edge in edges[('a', 'j', 'p')]:
        H1.append(edge[0])
        if mp.get(edge[1]) == None:
            mp[edge[1]] = tot
            tot += 1
        H2.append(mp[edge[1]])
    # for node in g2.nodes():
    #     H1.append(node)
    #     H2.append(tot)
    #     for i, neighborhood in enumerate(dgl.bfs_nodes_generator(g2, node)):
    #         if i >= 2:
    #             break
    #         for neighbor in neighborhood:
    #             H1.append(neighbor)
    #             H2.append(tot)
    #     tot += 1
    if len(H1) == 0:
        H1.append(0)
        H2.append(0)
    # H = dglsp.spmatrix(
    #     torch.LongTensor([H1, H2])
    # )
    H = torch.LongTensor([H1, H2])

    return H