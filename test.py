import torch
from model import MMRGNN

#
Fst = 5
Fs = 11
Ft = 1
N = 1276
M = 8
Fst_out = 1
embed_size = 10
H = 16
#
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
#
B = 64
S = 7
T = 1
xs = torch.randn((N, Fs))
xt = torch.randn((B, S, Ft))
yt = torch.randn((B, T, Ft))
xst = torch.randn((B, S, N, Fst))
edge_index = torch.randint(0, 1276, (2, 1275))
edge_weight = None
modality_index = torch.arange(N) % M
#
model = MMRGNN(
    Fs, Ft, Fst, N, M, Fst_out, embed_size, H, 
    augr_kwargs, enc_kwargs, mapper_kwargs, dec_kwargs, out_layer
)
model.debug = 1
yhat = model(
    xs=xs, xt=xt, xst=xst, yt=yt, T=T, 
    edge_index=edge_index, edge_weight=edge_weight, modality_index=modality_index
)
print(yhat.shape)
print(yhat)
