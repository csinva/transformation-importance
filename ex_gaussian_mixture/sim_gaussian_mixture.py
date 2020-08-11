import matplotlib.pyplot as plt
import numpy as np
import torch
import random
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import os,sys
opj = os.path.join
from tqdm import tqdm
import acd
from random import randint
from copy import deepcopy
from model import init_specific_model
from losses import Loss
from dset import *
from training import Trainer
from utils import *
import pickle as pkl

class p:
    '''Parameters for Gaussian mixture simulation
    '''
    # parameters for generating data
    train_n_samples_per_cluster = 10000
    test_n_samples_per_cluster = 2000
    latent_means = [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]]
    latent_vars = [[4, 2], [4, 2], [4, 2]]
    noise_dim = 8
    noise_var = 0.01
    
    # parameters for model architecture
    orig_dim = 2 + noise_dim
    latent_dim = 4    
    hidden_dim = orig_dim - 2
    
    # parameters for training
    train_batch_size = 64
    test_batch_size = 100
    lr = 1e-3
    beta = 0.0
    attr = 0.5
    alpha = 0.0
    gamma = 0.0
    tc = 0.0
    num_epochs = 50
    
    seed = 13
    
    # SAVE MODEL
    out_dir = "/home/ubuntu/transformation-importance/ex_gaussian_mixture/results"
    pid = ''.join(["%s" % randint(0, 9) for num in range(0, 20)])

    def _str(self):
        vals = vars(p)
        return 'beta=' + str(vals['beta']) + '_attr=' + str(vals['attr']) + '_tc=' + str(vals['tc']) + '_seed=' + str(vals['seed']) \
                + '_hdim=' + str(vals['hidden_dim']) + '_pid=' + vals['pid']
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
    
    
class s:
    '''Parameters to save
    '''
    reconstruction_loss = None
    kl_normal_loss = None
    disentanglement_metric = None
    total_correlation = None
    mutual_information = None
    dimensionwise_kl_loss = None
    attribution_loss = None
    net = None
    
    def _dict(self):
        return {attr: val for (attr, val) in vars(self).items()
                 if not attr.startswith('_')}
    
    
# generate data
def define_dataloaders(p):
    """A generic data loader
    """
    train_loader, train_latents = get_dataloaders(n_samples_per_cluster=p.train_n_samples_per_cluster, 
                                                  latent_means=p.latent_means,
                                                  latent_vars=p.latent_vars,
                                                  extra_dim=p.noise_dim, 
                                                  var=p.noise_var,
                                                  batch_size=p.train_batch_size,
                                                  shuffle=True,
                                                  return_latents=True) 
    test_loader, test_latents = get_dataloaders(n_samples_per_cluster=p.test_n_samples_per_cluster, 
                                                latent_means=p.latent_means,
                                                latent_vars=p.latent_vars,
                                                extra_dim=p.noise_dim, 
                                                var=p.noise_var,
                                                batch_size=p.test_batch_size, 
                                                shuffle=True,
                                                return_latents=True)   
    return((train_loader, train_latents), (test_loader, test_latents))


# calculate losses
def calc_losses(model, data_loader, loss_f):
        """
        Tests the model for one epoch.

        Parameters
        ----------
        data_loader: torch.utils.data.DataLoader

        loss_f: loss object

        Return
        ------
        """    
        model.eval()
        n_data = data_loader.dataset.data.shape[0]
        rec_loss = kl_loss = mi_loss = tc_loss = dw_kl_loss = attr_loss = 0
        
        for _, data in enumerate(data_loader):
            data = data.to(device)
            recon_data, latent_dist, latent_sample = model(data)
            _ = loss_f(data, recon_data, latent_dist, latent_sample, n_data) 
            rec_loss += loss_f.rec_loss.item()
            kl_loss += loss_f.kl_loss.item()
            mi_loss += loss_f.mi_loss.item()
            tc_loss += loss_f.tc_loss.item()
            dw_kl_loss += loss_f.dw_kl_loss.item()
            attr_loss += loss_f.attr_loss.item()     
        
        n_batch = len(data_loader)
        rec_loss /= n_batch
        kl_loss /= n_batch
        mi_loss /= n_batch
        tc_loss /= n_batch
        dw_kl_loss /= n_batch
        attr_loss /= n_batch
        
        return (rec_loss, kl_loss, mi_loss, tc_loss, dw_kl_loss, attr_loss)
    
    
def measure_anlge_iteration(model, data):
    batch_size, dim = data.shape
    
    results = []
    for batch_idx in range(batch_size):
        data_i = data[batch_idx:batch_idx+1]
        decoded_traversal = traversals(model, data=data_i, n_latents=p.latent_dim)[:,:2]
        
        variab = []
        for i in range(p.latent_dim):
            x = decoded_traversal[100*i:100*(i+1)]
            tot_var = torch.var(x[:,0]) + torch.var(x[:,1])
            variab.append(tot_var.item())
        variab = torch.Tensor(variab)
        _, idxs = torch.sort(variab, descending=True)
        idxs = idxs[:2]

        angles = []
        for i in range(2):
            x = decoded_traversal[100*idxs[i]:100*(idxs[i]+1)]
            v = x[-1] - x[0]
            angles.append(abs(v/torch.norm(v)))
        angles = torch.stack(angles)
        s1 = torch.sqrt((angles[0,0] - 1)**2 + (angles[1,1] - 1)**2)
        s2 = torch.sqrt((angles[0,1] - 1)**2 + (angles[1,0] - 1)**2)
        results.append(torch.min(s1, s2))

    return torch.stack(results)


def calc_disentangle_metric(model, data_loader):
    model.eval()
    
    dis_metric = []
    for _, data in enumerate(data_loader):
        results = measure_anlge_iteration(model, data)
        dis_metric.append(results)
        
    return torch.cat(dis_metric)


if __name__ == '__main__':
    # seed
    random.seed(p.seed)
    np.random.seed(p.seed)
    torch.manual_seed(p.seed)

    # GET DATALOADERS
    (train_loader, train_latents), (test_loader, test_latents) = define_dataloaders(p)

    # PREPARES MODEL
    model = init_specific_model(orig_dim=p.orig_dim, latent_dim=p.latent_dim, hidden_dim=p.hidden_dim)
    model = model.to(device)

    # TRAINS
    optimizer = torch.optim.Adam(model.parameters(), lr=p.lr)
    beta = p.beta
    attr = p.attr
    alpha = p.alpha
    gamma = p.gamma
    tc = p.tc
    num_epochs = p.num_epochs
    
    loss_f = Loss(beta=beta, attr=attr, alpha=alpha, gamma=gamma, tc=tc, is_mss=True)
    trainer = Trainer(model, optimizer, loss_f, device=device)
    trainer(train_loader, test_loader, epochs=num_epochs)
    
    # calculate losses
    print('calculating losses and metric...')    
    rec_loss, kl_loss, mi_loss, tc_loss, dw_kl_loss, attr_loss = calc_losses(model, test_loader, loss_f)
    s.reconstruction_loss = rec_loss
    s.kl_normal_loss = kl_loss
    s.total_correlation = tc_loss
    s.mutual_information = mi_loss
    s.dimensionwise_kl_loss = dw_kl_loss
    s.attribution_loss = attr_loss
    s.disentanglement_metric = calc_disentangle_metric(model, test_loader).mean()
    s.net = model    
    
    # save
    os.makedirs(p.out_dir, exist_ok=True)
    results = {**p._dict(p), **s._dict(s)}
    pkl.dump(results, open(opj(p.out_dir, p._str(p) + '.pkl'), 'wb'))    