
# Chi-Square Wavelet Graph Neural Networks for Heterogeneous Graph Anomaly Detection

## Tested Environment

- ubuntu 20.04
- python 3.9.13
- sklearn 1.0.2
- torch 2.6.0
- torchaudio 0.12.1
- torchvision 0.13.1
- numpy 1.24.4
- torch_geometric 2.6.1
- Scipy 1.9.1
- dgl 0.9.1.post1
- dgl-cu101 0.6.0
- cuda 10.1

## Datasets

We evaluated three heterogeneous datasets (ACM, R-I, and R-II) and seven homogeneous datasets. 

Among the **heterogeneous** datasets, only ACM is publicly accessible at https://drive.google.com/drive/folders/10-pf2ADCjq_kpJKFHHLHxr_czNNCJ3aX, where the files can be unzipped into the data/ directory. R-I and R-II are strictly closed source due to Tencent's privacy policy. 

For the **homogeneous** datasets, we directly adopt the data released by Tang et al. (2023), available at https://github.com/squareRoot3/GADBench/tree/master?tab=readme-ov-file.

**Reference**:
Tang, J., Hua, F., Gao, Z., et al. (2023). GADBench: Revisiting and benchmarking supervised graph anomaly detection. In *Advances in Neural Information Processing Systems* (Vol. 36, pp. 29628–29653).

**Directory Structure**

```
├── datas/
│   └── ACM/
│       ├── graph.dgl
│       ├── info.dat
│       ├── label.dat
│       ├── label.dat.test
│       ├── link.dat
│       └── node.dat
├── models/
│   ├── detector.py
│   └── gnn.py
├── benchmark.py
├── generate_edges.py
├── generate_graph.py
├── generate_id_mapping.py
├── random_walk_with_restart.py
├── set_seed.py
├── utils.py
└── run.sh
```

Use run.sh to train and test ChiGAD.

**Example**
```
chmod +x run.sh
run.sh
```

## Experiments

**Parameters**
- **seed**: Random seed for reproducibility, default = 3407
- **trials**: Number of trials for experiments, default = 1
- **model**: Model type to use, default = 'ChiGAD'
- **inductive**: Whether to use inductive learning, default = False
- **gpu**: GPU number to use, default = 0
- **dataset_name**: Dataset to use, default = 'ACM'
- **target_nt**: Target node type, default = '0'
- **q1**: Hyperparameter dividing Low and Mid Divisions, default = 1/3
- **q2**: Hyperparameter dividing Mid and High Divisions, default = 2/3
- **h_a**: Dimension size of aligned nodal representation, default = 512
- **mlp_layers**: Number of MLP layers, default = 4
- **lr**: Learning rate, default = 0.0001
- **dr_rate**: Dropout rate, default = 0.0
- **activation**: Activation function, default = 'ReLU'
- **H_**: Hyperparameter of upper bound of contribution-informed weight, default = 1.9/0.85
- **L_**: Hyperparameter of lower bound of contribution-informed weight, default = 1.9

**Example**
```
python benchmark.py --seed=3407 --trials=1 --model=ChiGAD --inductive= --gpu=0 --dataset_name=ACM --target_nt=0 --q1=0.3333 --q2=0.6667 --h_a=512 --mlp_layers=4 --lr=0.0001 --dr_rate=0 --activation=ReLU --H_=2.235 --L_=1.9
```

