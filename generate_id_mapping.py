import numpy as np
import torch
from tqdm import tqdm
import logging
import os
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

from set_seed import set_seed
set_seed()


def generate_id_info(dataset_name='ACM'):
    path_pre = 'datas/'
    file_name = 'node.dat'

    num_ntypes = 4      # For ACM dataset
    num_nodes = 10942   # For ACM dataset


    path_node = os.path.dirname(__file__) + '/' + path_pre + dataset_name + '/' + file_name
    id_mapping = {} # (from global_id to local_id)
    feat_mapping = {}# (from local_id to feat)
    dic_ntype_mask = {}# (from global_id to bool)
    dic_ntype = {}# (from global_id to ntype)

    local_i = np.zeros(num_ntypes).astype(int)  #4: num of ntypes


    with open(path_node, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip().split('\t')
            if len(line) == 4:  #4: there is feat
                global_id, content, ntype, feat = line
                global_id, ntype, feat = int(global_id), ntype, torch.tensor(np.array(feat.split(',')).astype(float))
            else:
                global_id, content, ntype = line
                global_id, ntype, feat = int(global_id), ntype, None

            #global id_mapping (from global_id to local_id)
            if global_id not in id_mapping.keys(): 
                id_mapping[global_id] = local_i[int(ntype)]
                local_i[int(ntype)] += 1

            #feat_mapping (from local_id to feat)
            if ntype not in feat_mapping.keys():
                feat_mapping[ntype] = {}
            if feat is not None:
                feat_mapping[ntype][id_mapping[global_id]] = feat

            #dic_ntype_mask (from global_id to bool)
            if ntype not in dic_ntype_mask.keys():
                dic_ntype_mask[ntype] = torch.zeros(num_nodes).bool()
            dic_ntype_mask[ntype][global_id] = True

            #dic_ntype (from global_id to ntype)
            if global_id not in dic_ntype.keys():
                dic_ntype[global_id] = ntype

            
    
    return id_mapping, feat_mapping, dic_ntype_mask, dic_ntype


