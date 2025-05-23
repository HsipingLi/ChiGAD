import numpy as np
import torch
import pickle
from tqdm import tqdm
import logging
import sys
import os
import warnings
import dgl
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
import random
from set_seed import set_seed
set_seed()

from generate_id_mapping import generate_id_info
from generate_edges import generate_edges



def generate_labels(target_nt='0', dataset_name='ACM'):
    id_mapping, feat_mapping, dic_ntype_mask, dic_ntype = generate_id_info(dataset_name)

    path_pre = 'datas/'
    ls_file_name = ['label.dat', 'label.dat.test']
    num_nodes = dic_ntype_mask[target_nt].sum() #only nodes of type 0 with labels
    labels = torch.zeros(num_nodes)  #3025:number of nodes
    ls_data = [] 
    for file_name in ls_file_name:
        path_labels = os.path.dirname(__file__) + '/' + path_pre + dataset_name + '/' + file_name
        with open(path_labels, 'r') as f:
            lines = f.readlines()
               # data: global_id, node_type, node_class
            for line in tqdm(lines):
                gu, ntype, nclass = np.array(line.strip().split()).astype(int)
                lu = id_mapping[gu]
                ls_data.append([lu, ntype, nclass])
    ls_data = np.array(ls_data)
    classes = np.unique(ls_data[:, -1])
    aclass = random.choice(classes)
    l_nodes = torch.tensor([id_mapping[u] for u in ls_data[:,0]])
    anodes = l_nodes[torch.tensor(ls_data[:, -1] == aclass)]
    labels[anodes] = 1

    return labels






def generate_mask(p_train=0.4, p_val=0.2, p_test=0.4, labels=None):

    num_tns = len(labels)   #tns: target nodes
    train_mask = torch.zeros(num_tns, dtype=torch.bool)
    val_mask = torch.zeros(num_tns, dtype=torch.bool)
    test_mask = torch.zeros(num_tns, dtype=torch.bool)

    indices_0 = (labels == 0).nonzero(as_tuple=True)[0]
    indices_1 = (labels == 1).nonzero(as_tuple=True)[0]

    num_train_0 = int(len(indices_0) * p_train) 
    num_val_0 = int(len(indices_0) * p_val)   
    num_test_0 = len(indices_0) - num_train_0 - num_val_0  

    num_train_1 = int(len(indices_1) * p_train) 
    num_val_1 = int(len(indices_1) * p_val)   
    num_test_1 = len(indices_1) - num_train_1 - num_val_1

    randperm_0 = torch.randperm(len(indices_0))
    train_indices_0 = indices_0[randperm_0[:num_train_0]]
    val_indices_0 = indices_0[randperm_0[num_train_0 : num_train_0+num_val_0]]
    test_indices_0 = indices_0[randperm_0[-num_test_0:]]

    randperm_1 = torch.randperm(len(indices_1))
    train_indices_1 = indices_1[randperm_1[:num_train_1]]
    val_indices_1 = indices_1[randperm_1[num_train_1 : num_train_1+num_val_1]]
    test_indices_1 = indices_1[randperm_1[-num_test_1:]]

    train_mask[train_indices_0] = 1
    train_mask[train_indices_1] = 1

    val_mask[val_indices_0] = 1
    val_mask[val_indices_1] = 1

    test_mask[test_indices_0] = 1
    test_mask[test_indices_1] = 1

    return train_mask, val_mask, test_mask






def generate_graph(target_nt = '0', inverse_direction = True, hetero_edge = True, dataset_name = "ACM"):


    path_pre = os.path.dirname(__file__) + '/datas/'



    path_graph = path_pre + dataset_name + '/graph.dgl'
    if os.path.exists(path_graph):
        return dgl.load_graphs(path_graph)[0][0]
    else:
        id_mapping, feat_mapping, dic_ntype_mask, dic_ntype = generate_id_info(dataset_name)
        edges_ = generate_edges(dataset_name)
        edges, r_edges = torch.tensor(edges_[:,:2].astype(int)), edges_[:, -1]


        node_types = list(dic_ntype_mask.keys())
        edge_types = np.unique(r_edges)
        
        het_edge_dict = {}
        if hetero_edge:
            for relation in edge_types:
                nt_u, nt_v = relation.split(' to ')
                relation = tuple([nt_u, relation, nt_v])
                het_edge_dict[relation] = (edges[:,0][r_edges == relation[1]], 
                                            edges[:,1][r_edges == relation[1]])

            if inverse_direction:
                for relation in edge_types:
                    nt_u, nt_v = relation.split(' to ')
                    inverse_link = nt_v + ' to ' + nt_u
                    inverse_relation = tuple([nt_v, inverse_link, nt_u])
                    if inverse_relation not in het_edge_dict:
                        het_edge_dict[inverse_relation] = (edges[:,1][r_edges == relation[1]], 
                                                    edges[:,0][r_edges == relation[1]])



        else:

            for relation in edge_types:
                homo_edge_raltion = list(relation)
                homo_edge_raltion[1] = '=='
                homo_edge_raltion = tuple(homo_edge_raltion)
                het_edge_dict[homo_edge_raltion] = (edges[:,0][edges[:,2] == et_mapping2[relation]], 
                        edges[:,1][edges[:,2] == et_mapping2[relation]])

                if inverse_direction:
                    inverse_homo_edge_raltion = homo_edge_raltion[::-1]
                    het_edge_dict[inverse_homo_edge_raltion] = (edges[:,1][edges[:,2] == et_mapping2[relation]], 
                        edges[:,0][edges[:,2] == et_mapping2[relation]])


        hg = dgl.heterograph(het_edge_dict)

        for nt in node_types:
            num_nodes_nt = dic_ntype_mask[nt].sum().item()
            if hg.num_nodes(nt) < num_nodes_nt:
                delta = num_nodes_nt - hg.num_nodes(nt)
                hg.add_nodes(delta, ntype = nt)
            if len(feat_mapping[nt]) > 0:
                raw_feats = torch.vstack(list(feat_mapping[nt].values()))
                hg.nodes[nt].data['feature'] = raw_feats


            else:
                hg.nodes[nt].data['feature'] = torch.randn(hg.num_nodes(nt), torch.randint(low = 1, high = 128, size = (1,)))

        label = generate_labels(target_nt, dataset_name)

        hg.nodes[target_nt].data['label'] = label



        ##################################################
        num_tnt = hg.num_nodes(target_nt)

        p_train = 0.4
        p_val = 0.2
        p_test = 0.4

        train_mask, val_mask, test_mask = generate_mask(p_train, p_val, p_test, label)

        hg.nodes[target_nt].data['train_mask'] = train_mask
        hg.nodes[target_nt].data['val_mask'] = val_mask
        hg.nodes[target_nt].data['test_mask'] = test_mask
        
        dgl.save_graphs(path_graph, hg)
                
        return hg
