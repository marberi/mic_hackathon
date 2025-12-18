# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: py4dstem
#     language: python
#     name: py4dstem
# ---

# %%
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm.notebook import tqdm
import hdbscan

import torch
from torch.utils.data import DataLoader

import data
import model


# %%

# %%
def load_network(ver, rot_loss):
    """Load the trained network.
       ver: Model version.
       rot_loss: If trained including the rotation loss. 
    """
    
    lbl = 'rotloss' if rot_loss else 'normal'
    net = model.ConvAutoencoder(in_channels=1, latent_ch=4, out_activation='none').cuda()
    
    state_dict = torch.load(d_models / f'model_angle_{lbl}_v{ver}.pt')
    net.load_state_dict(state_dict)
    
    return net


# %%
d_models = Path('/data/incaem/common/eriksen/hackathon/models')


# %%
# Location of the input data.
d = Path('/pnfs/pic.es/data/incaem/microscopy-hackathon/4dSTEMdata/1_1mrad_233nm_fov_256_256_12pA_CL460mm')
path = d / 'scan_x256_y256.raw'

# %%
ds = data.get_dataset(path).cuda()
loader = DataLoader(ds, batch_size=128, shuffle=False)


# %%
def find_features(net, loader):
    """Find the features predicted by the network.
       net: Auto-encoder network.
       loader: Data loader.
    """
    
    # By now also reconstructing the image
    net = net.eval()
    
    predL = []
    reconL = []
    for bimgs in tqdm(loader):
        with torch.no_grad():
            pred = net.enc(bimgs)
            
        tmp = pred.reshape((len(pred), -1))
        predL.append(tmp.cpu())
    
    comb = torch.cat(predL)

    return comb


# %%
def cluster_samples(inp, Nmin=60):
    """Cluster the feature vectors.
       inp: Input features.
       Nmin: Minimum cluster size in HDBScan.
    """
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=Nmin)
    cluster_id_map = clusterer.fit_predict(inp)
    cluster_id_map = cluster_id_map.reshape((256, 256))
    
    return cluster_id_map


# %%

# %%
ver = 4
net1 = load_network(ver, rot_loss=False)
net2 = load_network(ver, rot_loss=True)

# %%
feat1 = find_features(net1, loader)
feat2 = find_features(net2, loader)

# %%
cluster1 = cluster_samples(feat1)
cluster2 = cluster_samples(feat2)

# %%
fig, A = plt.subplots(ncols=2, sharey=True)
fig.set_size_inches([10, 4.5])

ax = A[0]
ax.imshow(cluster1, origin='lower')
ax.set_title('Default', fontsize=15)

ax = A[1]
ax.imshow(cluster2, origin='lower')
ax.set_title('Rotation-invariant latent constraint', fontsize=15)

plt.savefig('figs/clusters_v2.pdf')

# %%
