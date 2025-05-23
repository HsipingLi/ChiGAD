import subprocess
import sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])

import argparse
from utils import *
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")
import pickle
from set_seed import set_seed
import sys
from random_walk_with_restart import *
import logging
logging.basicConfig(level=logging.INFO)

from generate_graph import generate_graph

sys.path.append(os.path.dirname(__file__) + '/')

from tqdm import tqdm


def get_s_st(dic_ntsg, args):
    dic_s = {k:np.array([get_s_high(g, args.device) for g in dic_ntsg[k]]) for k in dic_ntsg.keys() if isinstance(dic_ntsg[k], list)}
    dic_st = {}
    for nt in dic_s.keys():
        s = dic_s[nt]
        st = np.zeros_like(s).astype(str)
        q1 = np.quantile(s, args.q1)
        q2 = np.quantile(s, args.q2)
        ind_low, ind_band, ind_high = s<=q1, (s>q1)&(s<=q2), s>q2
        st[ind_low], st[ind_band], st[ind_high] = 'low', 'band', 'high'
        dic_st[nt] = st
    return dic_s, dic_st



def compute_eigen(dic_det_g):

    dic = {}
    B = ['low', 'band', 'high']

    for ntype in dic_det_g.keys():
        dic_det_g_b = dic_det_g[ntype]
        for _, b in enumerate(tqdm(B)): 
            if _ == 0:
                dic[ntype] = {}
            g = dic_det_g_b[b]
            with g.local_scope():
                g = dgl.to_bidirected(g.add_self_loop())
                adj = g.adj().to_dense().numpy()
                adj = (adj!=0).astype(int)
                D = np.diag(np.sum(adj, axis = 1))
                D_inv_div = np.diag(1/(D**0.5))
                L = D_inv_div * (D- adj) * D_inv_div
                evalue, evec = np.linalg.eig(L)
            dic[ntype][b] = (evalue, evec)

    return dic




def compute_rep_lambds(dic_det_g, dic_eigen):

    dic_rep_lambs = {}

    B = ['low', 'band', 'high']

    for ntype in dic_det_g.keys():
        dic_det_g_b = dic_det_g[ntype]
        for b in B: 
            if b == 'low':
                dic_rep_lambs[ntype] = {}
            g = dic_det_g_b[b]
            evalue, evec = dic_eigen[ntype][b]
            with g.local_scope():
                feat = g.ndata['feature'].numpy()
                sorted_indices = np.argsort(evalue)

                lamb_ = np.real(evalue[sorted_indices])
                lamb_[lamb_<0] = 0
                u_ = np.real(evec[:, sorted_indices])
                weight = np.sum(u_@feat, axis=1)
                weight = weight**2/(weight**2).sum()


                bins = [0, 0.05, 0.3, 0.6, 0.9, 1.5, 2]

                K = len(bins)-1
                
                intervals = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
                interval_weights = np.zeros(K) 
                interval_midpoints = np.zeros(K)

                for i in range(K):
                    mask = (lamb_ >= bins[i]) & (lamb_ < bins[i+1])
                    interval_weights[i] = np.sum(weight[mask])
                    interval_midpoints[i] = np.median(lamb_[mask])

                mask = (lamb_ >= bins[-2]) & (lamb_ <= bins[-1])
                interval_weights[-1] = np.sum(weight[mask])
                interval_midpoints[-1] = np.median(lamb_[mask])

                lamb_ = interval_midpoints
                weight = interval_weights



                lamb_rep = lamb_[np.argmax(weight)]
                dic_rep_lambs[ntype][b] = lamb_rep

                print(f'nt: {ntype}, b: {b}')
                print(np.round(weight, 2))
                print(lamb_rep)
                print(intervals[np.argmax(weight)])
                print('*'*50)

                if b == 'high':
                    print('-'*50)
    return dic_rep_lambs




def load_or_compute_eigen(dic_det_g, file_path):

    file_path = os.path.expanduser(file_path)

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            dic_eigen = pickle.load(f)
    else:
        dic_eigen = compute_eigen(dic_det_g) 
        with open(file_path, 'wb') as f:
            pickle.dump(dic_eigen, f)
    
    return dic_eigen



def get_detected_g(dic_ntsg, dic_s, dic_st):
    dic_det_g = {}
    
    B = ['low', 'band', 'high']
    for nt in dic_st.keys():
        ntsg, s, st = dic_ntsg[nt], dic_s[nt], dic_st[nt]
        print()

        for _, b in enumerate(B):
            if _ == 0:
                dic_det_g[nt] = {}
            ind_B = np.where(st==b)[0]
            num = len(ind_B)
            med = s[ind_B][int(num/2)]
            idx = np.where(s==med)[0]
            med_g = ntsg[idx[0]]
            dic_det_g[nt][b] = med_g


    return dic_det_g



def assign_filter(dic_rep_lambs, final=[1,3,5,9]):
    B = ['low', 'band', 'high']
    map_assign = {
        0.0000: 1,
        0.6667: 2,
        1.1992: 4,
        1.5556: 8,
        1.7638: 16,
        1.8779: 32,
        1.9339: 64,
        1.9600: 128,
    }
    dic_assign = {}
    lamb_targ, filts = np.array(list(map_assign.keys())), np.array(list(map_assign.values()))
    for ntype in dic_rep_lambs.keys():
        for b in B:
            if b == 'low':
                dic_assign[ntype] = {}
            lamb_rep = dic_rep_lambs[ntype][b]

            filt = filts[np.argmin(np.abs(lamb_targ - lamb_rep))]
            dic_assign[ntype][b] = [0, filt]

    
    dic_assign['final'] = final

    print(dic_assign)
    return dic_assign








parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=3407)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--model', type=str, default='ChiGAD')
parser.add_argument('--inductive', type=bool, default=False)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset_name', type=str, default='ACM')
parser.add_argument('--target_nt', type=str, default='0')
parser.add_argument('--q1', type=float, default=1/3)
parser.add_argument('--q2', type=float, default=2/3)
parser.add_argument('--h_a', type=int, default=512)
parser.add_argument('--mlp_layers', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--dr_rate', type=float, default=0.0)
parser.add_argument('--activation', type=str, default='ReLU')
parser.add_argument('--H_', type=float, default=1.9/0.85)
parser.add_argument('--L_', type=float, default=1.9)

args = parser.parse_args()





args.device = torch.device(f"cuda:{args.gpu}")


model_result = {'name': args.model}
dataset_name = args.dataset_name
target_nt = args.target_nt

dic_ntsg = generate_ntsg_dic(dataset_name=dataset_name, target_nt=target_nt, iter = 10)
dic_g = dic_ntsg
dic_s, dic_st = get_s_st(dic_ntsg, args)




torch.cuda.empty_cache()
dic_det_g = get_detected_g(dic_ntsg, dic_s, dic_st)


file_path = '~/dic_eigen.pkl'
dic_eigen = load_or_compute_eigen(dic_det_g, file_path)
dic_rep_lambds = compute_rep_lambds(dic_det_g, dic_eigen)
filter_assigned = assign_filter(dic_rep_lambds, final=[1,3,5,9])





path_het_g = os.path.dirname(__file__) + '/datas/' + dataset_name + 'graph.dgl'
print(path_het_g)
if os.path.exists(path_het_g):
    het_g = dgl.load_graphs(path_het_g)[0][0]
else:
    het_g = generate_graph(dataset_name=dataset_name, target_nt=target_nt)


dic_g['het'] = het_g

ntypes = het_g.ntypes



graph = generate_ntsg_dic(dataset_name=dataset_name, target_nt=target_nt, iter=10, mpb=False)[target_nt]



dic_in_feats = {}
dic_h_feats = {}

for ntype in ntypes:
    in_feats = het_g.ndata['feature'][ntype].shape[1]
    dic_in_feats[ntype] = in_feats
    dic_h_feats[ntype] = args.h_a





train_config = {
    'device': args.device,
    'epochs': 200,
    'patience': 200,
    'metric': 'AUROC',
    'inductive': args.inductive
}

model_config = {
    'model': args.model,
    'lr': args.lr,
    'drop_rate': args.dr_rate,
    'mlp_layers': args.mlp_layers,    
    'activation': args.activation,
    'dic_h_feats': dic_h_feats,
    'h_a': args.h_a,
    'dic_g': dic_g,
    'ntypes': ntypes,
    'target_nt': target_nt,
    'dic_st': dic_st,
    'dic_coeff_mat': filter_assigned,
    'conv_type':'ChiGNN',
    'H_':args.H_,
    'L_':args.L_,
    'c_idx':7
}


auc_list, pre_list, precision_list, f1_list, recK_list = [], [], [], [], []
for t in range(args.trials):
    set_seed(args.seed)
    train_config['seed'] = args.seed
    if args.model == 'ChiGAD':
        detector = model_detector_dict[args.model](train_config=train_config, \
            model_config=model_config, graph=graph, dic_g=dic_g)
    else:
        try:
            detector = model_detector_dict[args.model](train_config=train_config, \
        model_config=model_config, graph=graph)
        except:
            detector = model_detector_dict[args.model](train_config=train_config, \
        model_config=model_config, data=graph)

    test_score = detector.train()
    auc_list.append(test_score['AUROC']), pre_list.append(test_score['AUPRC']), 
    precision_list.append(test_score['Precision']), f1_list.append(test_score['F1-macro']),
    recK_list.append(test_score['Recall'])
    del detector
    torch.cuda.empty_cache()

model_result[dataset_name+'-AUROC mean'] = np.mean(auc_list)
model_result[dataset_name+'-AUPRC mean'] = np.mean(pre_list)
model_result[dataset_name+'-Precision mean'] = np.mean(precision_list)
model_result[dataset_name+'-F1-macro mean'] = np.mean(f1_list)
model_result[dataset_name+'-Recall mean'] = np.mean(recK_list)

model_result = pd.DataFrame(model_result, index=[0])
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', None)     
pd.set_option('display.width', None)        

transposed_result = model_result.T

print(transposed_result)