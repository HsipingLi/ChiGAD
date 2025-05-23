import numpy as np
import torch
import pickle
from tqdm import tqdm
import logging
import sys
import os
import warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

from set_seed import set_seed
set_seed()

from generate_id_mapping import generate_id_info




def generate_edges(dataset_name='ACM'):


    id_mapping, feat_mapping, dic_ntype_mask, dic_ntype = generate_id_info(dataset_name)

    path_pre = 'datas/'
    file_name = 'link.dat'
    path_edges = os.path.dirname(__file__) + '/' + path_pre + dataset_name + '/' + file_name
    
    ls = []
    with open(path_edges, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = np.array(line.strip().split())
            gu, gv, _ = line[0:3].astype(int)
            r = dic_ntype[gu] + ' to ' + dic_ntype[gv]
            lu, lv = id_mapping[gu], id_mapping[gv]

            ls.append([lu, lv, r])

    edges = np.array(ls)
    return edges
