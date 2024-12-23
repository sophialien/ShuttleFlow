import torch
import numpy as np
from scipy.spatial import cKDTree as KDTree

from geomloss import SamplesLoss

def emd_approx(sample, ref):
    d = SamplesLoss(loss='sinkhorn', p=1, blur=0.01, backend='auto', debias=False)(sample, ref)
    return d

def distChamfer(a, b):
    x, y = torch.from_numpy(a), torch.from_numpy(b)
    num_x, _ = x.shape
    num_y, _ = y.shape
    xx = torch.mm(x, x.transpose(1, 0))
    yy = torch.mm(y, y.transpose(1, 0))
    zz = torch.mm(x, y.transpose(1, 0))
    diag_ind_x = torch.arange(0, num_x).long()
    diag_ind_y = torch.arange(0, num_y).long()
    rx = xx[diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz)
    ry = yy[diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz.transpose(1,0))
    P = (rx.transpose(1, 0) + ry - 2 * zz)

    return P.min(0)[0], P.min(1)[0]

def cal_CD(sample, reference):
    dl, dr = distChamfer(sample, reference)
    cd = dl.sum() + dr.sum()
    return cd

def distChamfer_batch(a, b):
    x, y = a, b #torch.from_numpy(a), torch.from_numpy(b)
    _, num_x, _ = x.shape
    _, num_y, _ = y.shape
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind_x = torch.arange(0, num_x).long()
    diag_ind_y = torch.arange(0, num_y).long()
    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(2).expand_as(zz)
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(2).expand_as(zz.transpose(2,1))
    P = (rx + ry.transpose(2, 1) - 2 * zz)
    return P.min(2)[0], P.min(1)[0]

def cal_CD_batch(sample, reference):
    dl, dr = distChamfer_batch(sample, reference)
    cd = dl.mean(dim=1) + dr.mean(dim=1)
    # cd = cd.cpu().detach().numpy()

    emd = emd_approx(sample, reference)
    # emd = emd.cpu().detach().numpy()

    return cd, emd

def multidimensional_KL(sample, reference):
    x, y = sample, reference
    n, d = x.shape
    m, _ = y.shape

    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the sample itself
    rx = xtree.query(x, k=2, eps=.01, p=2)[0][:,1] + 0.0001
    sx = ytree.query(x, k=1, eps=.01, p=2)[0]
    ry = ytree.query(y, k=2, eps=.01, p=2)[0][:,1] + 0.0001
    sy = xtree.query(y, k=1, eps=.01, p=2)[0]

    KL_XY = -np.log(sx/rx).sum() * d / n + np.log(m / (n - 1.))
    KL_YX = -np.log(sy/ry).sum() * d / m + np.log(n / (m - 1.))
    
    return KL_XY, KL_YX
