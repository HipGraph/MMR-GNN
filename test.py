import torch
import numpy as np

import util
import data
from model import MMRGNN


# Variables
## dataset vars
T = 31046 # total time-steps in the dataset
N = 1276 # number of nodes
Ti = 7 # time-steps into the model - history
To = 1 # time-steps forecast by the model - horizon
Fs = 11 # number of spatial features
Ft = 1 # number of temporal features
Fst = 5 # number of spatiotemporal features
Fst_out = 1 # number of forecasted spatiotemporal features - output size
## model vars
B = 64 # batch size
M = 8 # number of implicit modalities
embed_size = 10 # node embedding size
H = 16 # hidden size
augr_kwargs = {"graph_construction_method": ["top-k", ["dot", "Softmax"], 1.0]}
enc_kwargs = {
    "rnn_layer": "stGRU", 
    "rnn_kwargs": {
        "xs_size": embed_size, 
        "xt_size": Ft, 
        "conv": "cheb", 
        "layer": "kmLinear", 
        "order": 2, 
        "n_hops": 2, 
        "M": M, 
    }
}
mapper_kwargs = {"temporal_mapper": "last"}
dec_kwargs = dict(enc_kwargs)
dec_kwargs["rnn_kwargs"]["xs_size"] = 0
out_layer = "mLinear"
# Create data
## original data 
rng = np.random.default_rng(0)
spa = rng.normal(size=(N, Fs)) # spatial data - static node features
tmp = rng.normal(size=(T, Ft)) # temporal data - system-level dynamic features
spatmp = rng.normal(size=(T, N, Fst)) # spatiotemporal data - dynamic node features
edge_index = rng.integers(0, 1276, size=(2, 1275)) # graph edges in coo format
edge_weight = None # graph edge weights
## windowed data
in_index, out_index = util.input_output_window_indices(T, Ti, To)
xs = torch.tensor(spa, dtype=torch.float) # shape=(N, Fs)
xt = torch.tensor(tmp[in_index,:], dtype=torch.float)[:B,:,:] # shape=(B, Ti, Ft)
yt = torch.tensor(tmp[out_index,:], dtype=torch.float)[:B,:,:] # shape=(B, To, Ft)
xst = torch.tensor(spatmp[in_index,:,:], dtype=torch.float)[:B,:,:,:] # shape=(B, Ti, N, Fst)
edge_index = torch.tensor(edge_index, dtype=torch.long) # shape=(2, E)
hists, clustering, cluster_index = data.cluster(spatmp, "Agglomerative", M, "histogram", bins=12, lims=[-3,3])
modality_index = torch.tensor(cluster_index, dtype=torch.long) # shape=(N,)
# Init and run model
model = MMRGNN(
    Fs, Ft, Fst, N, M, Fst_out, embed_size, H, 
    augr_kwargs, enc_kwargs, mapper_kwargs, dec_kwargs, out_layer
)
model.debug = 1
yhat = model(
    xs=xs, xt=xt, xst=xst, yt=yt, T=To, 
    edge_index=edge_index, edge_weight=edge_weight, modality_index=modality_index
)
print(yhat.shape)
print(yhat)
