import os
import sys
sys.path.append(os.path.dirname(__file__) + '/')
sys.path.append(os.path.dirname(__file__) + '/models/')
import torch
import dgl.function as fn
import sympy as sp
import dgl.nn.pytorch.conv as dglnn
import dgl
from torch import nn
import numpy as np
from utils import *
from scipy.special import gamma
from scipy.integrate import quad


ls_S_high = []



def compute_laplacian_matrix(graph):


    graph = graph.cpu()
    num_nodes = graph.num_nodes()


    degrees = graph.out_degrees(range(num_nodes)).float()  
    D = torch.diag(degrees) 


    src, dst = graph.edges()
    A = torch.zeros(num_nodes, num_nodes)
    A[src, dst] = 1  

    L = D - A

    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-5))  
    L_norm = torch.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt

    return L, L_norm



def energy(graph, X):
    d = X.device
    dim = X.shape[1]
    L = compute_laplacian_matrix(graph)[1].to(d)
    X, L = X.float(), L.float()
    energy = ([(X[:,_]@L@X[:,_].unsqueeze(-1)/(X[:,_]@X[:,_].unsqueeze(-1))).detach().cpu().item() for _ in range(dim)])
    energy = np.mean(energy)
    return energy







class MLP(nn.Module):
    def __init__(self, in_feats, h_feats=256, num_classes=2, num_layers=2, drop_rate=0, activation='ReLU', **kwargs):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()
        if num_layers == 0:
            return
        if num_layers == 1:
            self.layers.append(nn.Linear(in_feats, num_classes))
        else:
            self.layers.append(nn.Linear(in_feats, h_feats))
            for i in range(1, num_layers-1):
                self.layers.append(nn.Linear(h_feats, h_feats))
            self.layers.append(nn.Linear(h_feats, num_classes))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, h, is_graph=True):
        if is_graph:
            h = h.ndata['feature']
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h)
            if i != len(self.layers)-1:
                h = self.act(h)
        return h




























class PolyConv(nn.Module):
    def __init__(self, theta):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)

    def forward(self, graph, feat):

        if isinstance(self._theta, list):
            self._theta = torch.tensor(np.array(self._theta).astype(float), dtype=torch.float32).to(feat.device)
        elif isinstance(self._theta, torch.Tensor):
            self._theta = torch.tensor(self._theta, dtype=torch.float32).to(feat.device)
        else:
            self._theta = torch.tensor(self._theta.astype(float), dtype=torch.float32).to(feat.device)


        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return feat - graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        return h




 

def calculate_theta_Chi(ls_d):
    thetas = []
    
    
    for d in ls_d:

        if d== 0:
            inv_coeff = [1]
        else:
            n = 2*d
            x = sp.symbols('x')
            i = n/2-1
            m = 1/(i+2)
            n, m = int(n), (m)
            
            from numpy.polynomial.chebyshev import chebfit, chebval


            y = np.vectorize(lambda x: (1/((2**(n/2))*gamma(n/2))) * ((x/m)**(n/2-1)) * np.exp(-(x/m)/2))
            S, _ = quad(y, 0, 2)
            
            lambs = np.linspace(0,2,500)
            cheb_coeff = chebfit(lambs, y(lambs)/S, 3)

            inv_coeff = cheb_coeff[::-1]


        thetas.append(inv_coeff)
    return thetas





class ChiGNN(nn.Module):
    def __init__(self, in_feats, h_feats=-1, num_classes=2, ls_d_coeff=[-1], mlp_layers=2, drop_rate=0,
                 activation='ReLU', **kwargs):
        super(ChiGNN, self).__init__()
        ls_d, ls_coeff = ls_d_coeff[0], ls_d_coeff[1]
        self.thetas = calculate_theta_Chi(ls_d)
        self.conv = []
        for i in range(len(self.thetas)):
            coeff = ls_coeff[i]
 
            self.conv.append(PolyConv(np.array(self.thetas[i]) * coeff))
        h_feats = int(1.2*in_feats) if h_feats==-1 else h_feats
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.mlp = MLP(h_feats*len(self.conv), h_feats, num_classes, mlp_layers, drop_rate)
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, graph):
        in_feat = graph.ndata['feature'].float()
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0], device=h.device)

        for conv in self.conv:
            h0 = conv(graph, h)
            h_final = torch.cat([h_final, h0], -1)
        h_final = self.dropout(h_final)
        h = self.mlp(h_final, False)
        return h









































def get_coeffs(coeff_mat):
    from collections import defaultdict
    for i in range(len(coeff_mat)-1):
        if i==0:
            poly1 = coeff_mat[0]
        else:
            poly1 = ls_d
        poly2 = coeff_mat[i+1]

        poly1_dict = defaultdict(int)
        poly2_dict = defaultdict(int)
        for term in poly1:
            poly1_dict[term] = 1
        for term in poly2:
            poly2_dict[term] = 1
        
        degrees = set()
        for degree1 in poly1_dict:
            for degree2 in poly2_dict:
                degrees.add(degree1 + degree2)

        ls_d = sorted(list(degrees))
    
    return ls_d




def filter_conv(dic, w_decay=1/3):
    dic_new = {}
    ntypes, B = dic.keys(), ['low', 'band', 'high']
    for nt in ntypes:
        for idx_b, b in enumerate(B):

            if b == 'low':
                dic_new[nt] = {}

            polynomials = list(dic[nt].values())
            result = np.polynomial.Polynomial([1])
            for idx_d, degrees in enumerate(polynomials):
                coeffs = [0] * (max(degrees) + 1) 
                for d in degrees:
                    coeffs[d] = 1 if (idx_d==idx_b or d==0) else w_decay
                poly = np.polynomial.Polynomial(coeffs) 
                result *= poly

            degrees = [i for i, coeff in enumerate(result.coef) if coeff != 0]
            coeffs = np.round([coeff[1] for coeff in enumerate(result.coef) if coeff[1] != 0],4)
            dic_new[nt][b] = [degrees, coeffs]


    return dic_new





class ChiGAD(nn.Module):       
    def __init__(self, dic_g,
                    dic_h_feats, mlp_layers, drop_rate, activation, 
                        h_a, ntypes=None, target_nt='0',
                            **kwargs):
        super(ChiGAD, self).__init__()
        
        self.conv_type = kwargs['conv_type']

        self._ = 0
        self.ntypes = ntypes
        self.tnt = target_nt
        self.dic_g = dic_g

        self.homoMapping = self.get_Homo_Mapping(dic_g)
        self.dic_g['homo'] = self.get_Homo_Graph(dic_g, self.homoMapping)

        dic_in_feats = {ntype: dic_g[ntype][0].ndata['feature'].shape[1] for ntype in ntypes}


        
        self.dic_st = kwargs['dic_st']
        dic_coeff_mat = kwargs['dic_coeff_mat'] #l b h: low, band, high
        self.coeff_mat_node_wise, self.coeff_mat_final \
            = {k: dic_coeff_mat[k] for k in self.ntypes},\
                dic_coeff_mat['final']
        
        self.coeff_mat_node_wise = filter_conv(self.coeff_mat_node_wise)
        self.coeff_mat_final = [self.coeff_mat_final, np.ones_like(self.coeff_mat_final)]
        

        self.mlpModule = self.construct_MLP_Module(dic_g=self.dic_g, dic_h_feats=dic_h_feats, ntypes=self.ntypes, \
            mlp_layers=mlp_layers, drop_rate=drop_rate, activation=activation)
    
        self.h_a = h_a

        self.homoModule = self.construct_Homo_Module(dic_in_feats, dic_h_feats, mlp_layers, drop_rate, activation, conv_type=self.conv_type)    
        self.global_specGNN = ChiGNN(h_a, int(h_a*1.2), 2, self.coeff_mat_final, mlp_layers, drop_rate, activation)
        

    def get_Homo_Mapping(self, dic_g):
        dic_homo_mapping = {}
        het_g = dic_g['het']
        list_num_nodes = list(het_g.num_nodes(ntype) for ntype in self.ntypes)
        for idx, ntype in enumerate(self.ntypes):
            dic_homo_mapping[ntype] = sum(list_num_nodes[:idx])
        
        return dic_homo_mapping



    def get_Homo_Graph(self, dic_g, dic_homo_mapping):
        het_g = dic_g['het']

        def ntype_map(ntype):
            if ntype not in self.ntypes:
                return 'uin'
            else:
                return ntype
        tnt_mask = torch.zeros(het_g.num_nodes()).bool()
        for etype in het_g.etypes:
            type_uv = etype.split(' to ')
            type_u, type_v = ntype_map(type_uv[0]), ntype_map(type_uv[1])
            add_u, add_v = dic_homo_mapping[type_u], dic_homo_mapping[type_v]
            het_src, het_dst = het_g.edges(etype=etype)
            homo_src, homo_dst = het_src + add_u, het_dst + add_v
            if het_g.etypes.index(etype) == 0:
                src, dst = homo_src, homo_dst
            else:
                src, dst = torch.cat((src, homo_src)), torch.cat((dst, homo_dst))
            if type_u == self.tnt: tnt_mask[homo_src] = True
            if type_v == self.tnt: tnt_mask[homo_dst] = True
        
        homo_g = dgl.graph((src, dst), num_nodes=het_g.num_nodes())
        homo_g.ndata['tnt_mask'] = tnt_mask
        return homo_g



    def homoSpecLayers(self, in_feats, h_feats, ls_d_coeff, mlp_layers, drop_rate,
                 activation='ReLU', conv_type='ChiGNN'):
        
        h_output = h_feats
        SpecLayers = ChiGNN(in_feats, h_feats, h_output, ls_d_coeff, mlp_layers, drop_rate, activation)

        return SpecLayers
        

    def construct_Homo_Module(self, dic_in_feats, dic_h_feats, mlp_layers, drop_rate,
                  activation, conv_type='ChiGNN'):

        homoModule = {}
        for idx, ntype in enumerate(self.ntypes):
            in_feats, h_feats = dic_in_feats[ntype], dic_h_feats[ntype]
            homoModule[ntype] = {}
            SpecLayers_l = self.homoSpecLayers(in_feats, h_feats, self.coeff_mat_node_wise[ntype]['low'], mlp_layers, drop_rate, activation, conv_type)
            SpecLayers_b = self.homoSpecLayers(in_feats, h_feats, self.coeff_mat_node_wise[ntype]['band'], mlp_layers, drop_rate, activation, conv_type)
            SpecLayers_h = self.homoSpecLayers(in_feats, h_feats, self.coeff_mat_node_wise[ntype]['high'], mlp_layers, drop_rate, activation, conv_type)
            homoModule[ntype]['low'], homoModule[ntype]['band'], homoModule[ntype]['high']= SpecLayers_l, SpecLayers_b, SpecLayers_h

            for layer in list(homoModule[ntype].values()):
                for param in layer.parameters():
                    self.register_parameter(str(self._), param)
                    self._ += 1


        return homoModule




    def construct_MLP_Module(self, dic_g, dic_h_feats, ntypes, mlp_layers, drop_rate, activation):
        dic_MLP = {}
        for nt in ntypes:
            h_feats = dic_h_feats[nt]
            num_g = len(dic_g[nt])
            dic_MLP[nt] = MLP(num_g*h_feats, h_feats, h_feats, mlp_layers, drop_rate, activation)


        for layer in list(dic_MLP.values()):
            for param in layer.parameters():
                self.register_parameter(str(self._), param)
                self._ += 1

        return dic_MLP







    def forward(self, train_graph):

        dic_gs_aligned = {}
        for idx, ntype in enumerate(self.ntypes):

            h_final = torch.zeros([self.dic_g[ntype][0].num_nodes(), 0], device=train_graph.device)
            homo_gs = [g.to(list(self.parameters())[0].device) for g in self.dic_g[ntype]]
            for i, homo_g in enumerate(homo_gs):         
                st = self.dic_st[ntype][i]  #st: S_high type (low, band, high)  
                h0  = self.homoModule[ntype][st](homo_g)
                h_final = torch.cat([h_final, h0], -1)

        
            # align with linear layer    
            gs_aligned = self.mlpModule[ntype](h_final, False)
            dic_gs_aligned[ntype] = gs_aligned

          

        # '''
        # global specGNN
        homo_g = self.dic_g['homo'].to(list(self.parameters())[0].device)

        with homo_g.local_scope():
            h = torch.cat([dic_gs_aligned[ntype] for ntype in self.ntypes])
            # h = torch.cat([dic_gs_aligned[ntype] for ntype in self.ntypes])
            homo_g.ndata['feature'] = h


        

            gs_final = self.global_specGNN(homo_g)

            # gs_final = self.mlp(gs_final, False)
            tnt_mask = homo_g.ndata['tnt_mask']
            h = gs_final[tnt_mask]

        return h



