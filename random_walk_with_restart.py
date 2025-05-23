import torch
import numpy as np
import pickle
from tqdm import tqdm
from dgl.sampling import random_walk
import dgl
import os
import random
from set_seed import set_seed
set_seed()
import sys
pp = os.path.dirname(os.path.dirname(__file__))
import logging 
logging.basicConfig(level=logging.INFO)
from generate_graph import generate_graph




def get_s_high(g, device = torch.device('cpu')):


    with g.local_scope():
        if isinstance(g, dgl.DGLGraph):
            if len(g.etypes) <= 1 and 'feature' in g.ndata.keys():
                g = g.to(device)
                num_nodes = g.num_nodes()
                A = torch.zeros((num_nodes, num_nodes)).to(device)
                A[g.edges()[0], g.edges()[1]] = 1
                diag = torch.sum(A, dim=1).to(device)
                D = torch.diag(diag)
                L = D - A
                feat = g.ndata['feature'].float()
                s_high = 0
                for i in range(feat.shape[1]):
                    x = feat[:, i].unsqueeze(0)
                    s_high += ((x@L@x.T)/(x@x.T))[0].item()
                s_high = s_high/feat.shape[1]
    del g, A, D, L
    return s_high










def generate_meta_paths(lens=None, ntypes=None, etypes=None):
    import itertools
    dic_mp = {} #mp: meta path

    def valid(p):
            val_concat = np.min([p[i].split(' to ')[1]==p[i+1].split(' to ')[0] for i in range(len(p)-1)])
            val_h_t = p[0].split(' to ')[0] == p[-1].split(' to ')[1]   #h t: heda and tail
            return val_concat & val_h_t
    lens = np.array(lens).astype(int)
    for len_ in lens:
        ps  = tuple(itertools.product(etypes, repeat=len_)) #ps: paths
        for p in ps:    #p: path
            if valid(p):
                nt = p[0].split(' to ')[0]  #nt: node type
                if nt not in dic_mp.keys():
                    dic_mp[nt] = {p}
                else:
                    dic_mp[nt].add(p)
    return dic_mp








def rwr_mpb(path_g, target_nt, iter=1): #mpb: meta path based random walk
    dic_ntsg = {}
    if os.path.exists(path_g):
        graph = dgl.load_graphs(path_g)[0][0]
    else:
        graph = generate_graph(target_nt=target_nt)

    dic_mp = generate_meta_paths(lens=np.arange(2,6).astype(int), ntypes=graph.ntypes, etypes=graph.etypes)


    for ntype in graph.ntypes:
        # ls_src, ls_dst = [], [] 
        
        for mps in dic_mp[ntype]: #mps: meta paths
            
            for _ in range(iter):
                walks = random_walk(graph, graph.nodes(ntype), restart_prob=1e-100, metapath = mps)[0]
                src, dst = walks[:,0], walks[:,-1]
                try:
                    valid_indices = (src>=0) & (dst>=0)
                except:
                    logging.info(f'{src}, {src>=0}')
                src, dst = src[valid_indices], dst[valid_indices]

        
            nt_sg = dgl.graph((src, dst), num_nodes=graph.num_nodes(ntype))
            nt_sg.ndata['feature'] = graph.ndata['feature'][ntype]
            if ntype == target_nt:
                nt_sg.ndata['label'] = graph.ndata['label'][ntype]
                nt_sg.ndata['train_mask'] = graph.ndata['train_mask'][ntype]
                nt_sg.ndata['val_mask'] = graph.ndata['val_mask'][ntype]
                nt_sg.ndata['test_mask'] = graph.ndata['test_mask'][ntype]

            if ntype not in dic_ntsg.keys():
                dic_ntsg[ntype] = {nt_sg}
            else:
                dic_ntsg[ntype].add(nt_sg)

    for k in dic_ntsg.keys(): dic_ntsg[k] = list(dic_ntsg[k])

    dic_ntsg['iter'] = iter

    return dic_ntsg















def rwr(path_g, target_nt, iter=1):
    dic_ntsg = {}
    if os.path.exists(path_g):
        graph = dgl.load_graphs(path_g)[0][0]
    else:
        graph = generate_graph(target_nt=target_nt)

    for ntype in graph.ntypes:
        ls_src, ls_dst = [], [] 
        for etype in graph.etypes:
            etype_split = etype.split(' ')
            src_type = etype_split[0]
            dst_type = etype_split[2]

            if ntype == src_type:
                inv_etype = etype.split(' ')
                inv_etype = inv_etype[2] + ' to ' + inv_etype[0]
                meta_path = [etype, inv_etype]
                for _ in range(iter):
                    walks = random_walk(graph, graph.nodes(ntype), restart_prob=1e-100, metapath = meta_path)[0]
                    src, dst = walks[:,0], walks[:,2]
                    try:
                        valid_indices = (src>=0) & (dst>=0)
                    except:
                        logging.info(f'{src}, {src>=0}')
                    src, dst = src[valid_indices], dst[valid_indices]
                    ls_src.append(src), ls_dst.append(dst)

            elif ntype == dst_type:
                inv_etype = etype.split(' ')
                inv_etype = inv_etype[2] + ' to ' + inv_etype[0]
                meta_path = [inv_etype, etype]
                for _ in range(iter):
                    walks = random_walk(graph, graph.nodes(ntype), restart_prob=1e-100, metapath = meta_path)[0]
                    src, dst = walks[:,0], walks[:,2]
                    try:
                        valid_indices = (src>=0) & (dst>=0)
                    except:
                        logging.info(f'{src}, {src>=0}')
                    src, dst = src[valid_indices], dst[valid_indices]
                    ls_src.append(src), ls_dst.append(dst)

            else:
                continue
        
        src, dst = torch.cat(ls_src), torch.cat(ls_dst)
        nt_sg = dgl.graph((src, dst), num_nodes=graph.num_nodes(ntype))
        nt_sg.ndata['feature'] = graph.ndata['feature'][ntype]
        if ntype == target_nt:
            nt_sg.ndata['label'] = graph.ndata['label'][ntype]
            nt_sg.ndata['train_mask'] = graph.ndata['train_mask'][ntype]
            nt_sg.ndata['val_mask'] = graph.ndata['val_mask'][ntype]
            nt_sg.ndata['test_mask'] = graph.ndata['test_mask'][ntype]
        dic_ntsg[ntype] = nt_sg
    
    dic_ntsg['iter'] = iter

    return dic_ntsg






def generate_ntsg_dic(dataset_name='ACM', target_nt='0', iter = 1, mpb = True):

    path_g = os.path.dirname(__file__) + '/datas/' + dataset_name + '/graph.dgl'
    if mpb:
        dic_ntsg = rwr_mpb(path_g, target_nt, iter)
    else:
        dic_ntsg = rwr(path_g, target_nt, iter)

    return dic_ntsg



