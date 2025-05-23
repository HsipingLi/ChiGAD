from sklearn import svm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score
import dgl
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(__file__) + '/')
sys.path.append(os.path.dirname(__file__) + '/models/')
from models.gnn import *
import logging
logging.basicConfig(level=logging.INFO)
import torch.nn.functional as F





def find_best_f1(probs, labels):
    best_f1, best_thre = -1., -1.
    thres_arr = np.linspace(0.05, 0.95, 100)
    for thres in thres_arr:
        preds = np.zeros(len(labels))
        try:
            probs, labels = probs.cpu(), labels.cpu()
        except:
            pass
        preds[probs > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre





class BaseDetector(object):
    def __init__(self, train_config, model_config, graph, dic_g = None):
        self.model_config = model_config
        self.train_config = train_config
        self.graph = graph

        model_config['in_feats'] = self.graph.ndata['feature'].shape[1]
        graph = self.graph.to(self.train_config['device'])
        if dic_g is not None:
            self.dic_g = {}
            for g in dic_g.keys():
                try:
                    self.dic_g[g] = dic_g[g].to(self.train_config['device'])
                except:
                    continue

        self.labels = graph.ndata['label']
        self.train_mask = graph.ndata['train_mask'].bool()
        self.val_mask = graph.ndata['val_mask'].bool()
        self.test_mask = graph.ndata['test_mask'].bool()
        self.weight = (1 - self.labels[self.train_mask]).sum().item() / self.labels[self.train_mask].sum().item()
        
        
        
        if self.model_config['model'] in ['ChiGAD', 'ChiGNN']:
            self.H_, self.L_ = self.model_config['H_'], self.model_config['L_']

        
        
        self.source_graph = graph
        try:
            print(train_config['inductive'])
            if train_config['inductive'] == False:
                self.train_graph = graph
                self.val_graph = graph
            else:
                self.train_graph = graph.subgraph(self.train_mask)
                self.val_graph = graph.subgraph(self.train_mask+self.val_mask)
        except:
            self.train_graph = graph
            self.val_graph = graph
        self.best_score = -1
        self.patience_knt = 0
        
    def train(self):
        pass



    def eval(self, labels, probs, labels_hat):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()
            if torch.is_tensor(labels_hat):
                labels_hat = labels_hat.cpu().numpy()



            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)


            score['Precision'] = precision_score(labels, labels_hat)
            score['F1-macro'] = f1_score(labels, labels_hat, average='macro')
            

            labels = np.array(labels)
            k = int(labels.sum())
            score['Recall'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)
        return score




    def eval2(self, labels, probs, thre=None):
        score = {}
        with torch.no_grad():
            if torch.is_tensor(labels):
                labels = labels.cpu().numpy()
            if torch.is_tensor(probs):
                probs = probs.cpu().numpy()

            score['AUROC'] = roc_auc_score(labels, probs)
            score['AUPRC'] = average_precision_score(labels, probs)


            score['Precision'] = -1


            labels = np.array(labels)
            k = int(labels.sum())
            score['Recall'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)

        # F1-macro
            if thre is not None:
                labels_hat = torch.zeros(labels.shape[0])
                labels_hat[probs>thre]=1
                score['F1-macro'] = f1_score(labels, labels_hat, average='macro')
            else:
                score['F1-macro'] = -1
        return score


class BaseGNNDetector(BaseDetector):
    def __init__(self, train_config, model_config, graph, dic_g = None):
        super().__init__(train_config, model_config, graph, dic_g)
        gnn = globals()[model_config['model']]
        try:
            model_config['in_feats'] = self.graph.ndata['feature'].shape[1]
        except:
            model_config['in_feats'] = self.graph.ndata['feature'].shape[1]

        if model_config['model'] == 'ChiGAD':
            model_config_cp = model_config.copy()
            if 'dic_g' not in model_config_cp.keys():
                model_config_cp['dic_g'] = dic_g
            model_config_cp['num_layers'] = -1
            self.model = gnn(**model_config_cp).to(train_config['device'])
        else:
            self.model = gnn(**model_config).to(train_config['device'])




    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        train_labels, val_labels, test_labels = self.labels[self.train_mask], self.labels[self.val_mask], self.labels[self.test_mask]
        
        
        best_metric = -1
        best_epoch = -1
        best_dic = {}
        


        from dgl.nn import GraphConv
        g = dgl.add_self_loop(self.train_graph)

        conv = GraphConv(1, 1, norm='none', weight=False, bias=False)
        L = lambda X: ((((g.in_degrees()*X.T) - conv(g,X).T).T)/g.in_degrees().unsqueeze(1))


        ls_c = []
        feat = g.ndata['feature'].clone()
        for _ in range(8):  # contributions of representations of node of target type
            c = 0  #contributions
            for i in range(feat.shape[1]):
                x = feat[:,i].unsqueeze(1)
                Lx = L(x)
                S = torch.sum(x*Lx)
                c = c + torch.sum(x*Lx, axis=1)/S
                feat[:,i] = Lx.T[0]
            ls_c.append(c)         



        indices_0, indices_1 = torch.where(train_labels==0)[0], torch.where(train_labels==1)[0]
        con = ls_c[self.model_config['c_idx']][self.train_mask]
        c_max, c_min = con[indices_1].max(), con[indices_1].min()

        weight = (c_max - con)/(c_max - c_min)*(self.H_-self.L_) + self.L_                    #official normalization method (min-max normalization)
        weight = ((self.H_-self.L_)/(torch.exp(c_max)-1)*(torch.exp(c_max-con)-1)+self.L_)      #alternative normalization method (nonlinear normalization)

        weight[indices_0] = 1

        for e in range(self.train_config['epochs']):
            self.model.train()

            logits = self.model(self.train_graph)

            
            logits_train = logits[self.train_graph.ndata['train_mask']]

            indices = torch.tensor(np.arange(train_labels.shape[0])).to(self.labels.device)
            ce_loss = -F.log_softmax(logits_train, dim=1)[indices,train_labels.long()]
            
            ce_loss = ce_loss * weight
            ce_loss = ce_loss.mean()

            loss = ce_loss
  
            
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            probs = logits.softmax(1)[:, 1]


            if self.model_config['drop_rate'] > 0 or self.train_config['inductive']:
                self.model.eval()
                logits = self.model(self.val_graph)
            probs = logits.softmax(1)[:, 1]
            y_pred = torch.argmax(logits.softmax(1), dim=1)
            _, v_thre = find_best_f1(probs[self.val_graph.ndata['val_mask']], val_labels)
            
            
            
                        
            label_t, probs_t = self.labels[self.val_mask+self.test_mask], probs[self.val_mask+self.test_mask]
            pred_t = torch.zeros_like(label_t); pred_t[probs_t>v_thre] = 1
            
            FN_ind, FP_ind = (pred_t==0) & (label_t==1), (pred_t==1) & (label_t==0)
            TP_ind, TN_ind = (pred_t==1) & (label_t==1), (pred_t==0) & (label_t==0)
            FN = (FN_ind).sum()
            FP = (FP_ind).sum()

            TP = (TN_ind).sum()
            TN = (TP_ind).sum()
            


            print(FN, FP)
            

            
            
            val_score = self.eval(val_labels, probs[self.val_graph.ndata['val_mask']], y_pred[self.val_graph.ndata['val_mask']])
            if val_score[self.train_config['metric']] >= self.best_score:
                if self.train_config['inductive']:
                    logits = self.model(self.source_graph)
                    probs = logits.softmax(1)[:, 1]

                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval2(test_labels, probs[self.test_mask], v_thre)

            self.patience_knt += 1
            if self.patience_knt > self.train_config['patience']:
                break


            test_score = self.eval2(test_labels, probs[self.test_mask], v_thre)
            logging.info(f"Epoch {e}: " + str(test_score))

            
            print('Epoch {}, Loss {:.4f}\nVal AUC {:.4f}, PRC {:.4f}, Precision {:.4f}, F1-macro {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['Precision'], val_score['F1-macro']))
            print('-'*40)


            if test_score[self.train_config['metric']] > best_metric:
                best_dic = test_score.copy()
                best_metric = test_score[self.train_config['metric']]
                best_epoch = e
            
            print('Best Epoch {}, \ntest AUC {:.6f}, PRC {:.6f}, Precision {:.6f}, F1-macro {:.6f}, Recall {:.6f}'.format(
                    best_epoch, best_dic['AUROC'], best_dic['AUPRC'], best_dic['Precision'], best_dic['F1-macro'],  best_dic['Recall']))

        
        return test_score


