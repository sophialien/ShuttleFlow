from pprint import pprint
from sklearn.svm import LinearSVC
from math import log, pi
import os
import torch
import torch.distributed as dist
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import ndimage
import imageio

from sklearn.neighbors import NearestNeighbors
from metrics.JSD import jsd_between_list_of_points, grid_coordinate
from metrics.CD_EMD import cal_CD, multidimensional_KL, cal_CD_batch

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2 * pi)
    b = logvar
    c = (x - mean) ** 2 / torch.exp(logvar)
    return -0.5 * torch.sum(a + b + c)


def bernoulli_log_likelihood(x, p, clip=True, eps=1e-6):
    if clip:
        p = torch.clamp(p, min=eps, max=1 - eps)
    return torch.sum((x * torch.log(p)) + ((1 - x) * torch.log(1 - p)))


def kl_diagnormal_stdnormal(mean, logvar):
    a = mean ** 2
    b = torch.exp(logvar)
    c = -1
    d = -logvar
    return 0.5 * torch.sum(a + b + c + d)


def kl_diagnormal_diagnormal(q_mean, q_logvar, p_mean, p_logvar):
    # Ensure correct shapes since no numpy broadcasting yet
    p_mean = p_mean.expand_as(q_mean)
    p_logvar = p_logvar.expand_as(q_logvar)

    a = p_logvar
    b = - 1
    c = - q_logvar
    d = ((q_mean - p_mean) ** 2 + torch.exp(q_logvar)) / torch.exp(p_logvar)
    return 0.5 * torch.sum(a + b + c + d)


# Taken from https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
def truncated_normal(tensor, mean=0, std=1, trunc_std=2):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < trunc_std) & (tmp > -trunc_std)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def reduce_tensor(tensor, world_size=None):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if world_size is None:
        world_size = dist.get_world_size()

    rt /= world_size
    return rt


def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2


def set_random_seed(seed):
    """set random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Visualization pts->generated, gtr->ground truth
def visualize_hit_points(pts, gtr, idx, cls, m, s, space):
    pts = pts.cpu().detach().numpy()
    gtr = gtr.cpu().detach().numpy()

    gtr = gtr * s + m
    pts = pts * s + m

    fig = plt.figure(figsize=(5, 3))
    ax1 = fig.add_subplot(121)
    ax1.set_title("Sample:%s, c=%s" % (idx, cls))
    ax1.scatter(pts[:, 0], pts[:, 1], s=5, c='blue', alpha=0.2)
    ax1.set_xlim([25, 325]) # (25, 325)
    ax1.set_ylim([480, 150]) # (150, 480)
    h_box = 330//space['h_space']
    for y in range(1, h_box):
        line = mlines.Line2D([25, 325], [150+y*space['h_space'], 150+y*space['h_space']], color='grey', alpha=0.2)
        ax1.add_line(line)
    w_box = 300//space['w_space']
    for x in range(1, w_box):
        line = mlines.Line2D([25+x*space['w_space'], 25+x*space['w_space']], [150, 480], color='grey', alpha=0.2)
        ax1.add_line(line)

    ax2 = fig.add_subplot(122)
    ax2.set_title("Ground Truth:%s, c=%s" % (idx, cls))
    ax2.scatter(gtr[:, 0], gtr[:, 1], s=5, c='blue', alpha=0.2)
    ax2.set_xlim([25, 325]) # (25, 325)
    ax2.set_ylim([480, 150]) # (150, 480)
    h_box = 330//space['h_space']
    for y in range(1, h_box):
        line = mlines.Line2D([25, 325], [150+y*space['h_space'], 150+y*space['h_space']], color='grey', alpha=0.2)
        ax2.add_line(line)
    w_box = 300//space['w_space']
    for x in range(1, w_box):
        line = mlines.Line2D([25+x*space['w_space'], 25+x*space['w_space']], [150, 480], color='grey', alpha=0.2)
        ax2.add_line(line)
    
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    res = np.array(fig.canvas.renderer._renderer)
    res = np.transpose(res, (2, 0, 1))

    plt.close()
    return res

def draw_heatmap(pts, s, nn):
    _, indices = nn.kneighbors(pts)
    indices = np.squeeze(indices)

    grid_counters = np.zeros(660*600)
    for i in indices:
        grid_counters[i] += 1
    
    heatmap = grid_counters.reshape(660, 600)

    heatmap = ndimage.filters.gaussian_filter(heatmap, sigma=s, mode='constant')
    heatmap = heatmap/heatmap.sum()

    return heatmap

def heatmap_MSE(r_points, g_points, nn, sigma=30):
    r_points = r_points
    g_points = g_points

    h_real = draw_heatmap(r_points, sigma, nn)
    h_gen = draw_heatmap(g_points, sigma, nn)
    minus = (h_real - h_gen)

    return np.sqrt(np.sum(minus*minus))


def save(model, optimizer, epoch, path):
    d = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(d, path)


def resume(path, model, optimizer=None, strict=True):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'], strict=strict)
    start_epoch = ckpt['epoch']
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    return model, optimizer, start_epoch


def validate(test_loader, model, epoch, writer, save_dir, args, mean, std, tr_val):
    model.eval()

    space = {'w_space': args.w_space, 'h_space': args.h_space}
    grid_coordinates = grid_coordinate(0.5, 0.5)
    grid_coordinates = grid_coordinates.reshape(-1, 2)
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    MSE = []
    JSD = []
    CD = []
    EMD = []
    for bidx, data in enumerate(test_loader):
        idx_batch, points, cons, playerX, playerY, points_all, num_points = \
                    data['idx'], data['points'], data['cons'], data['playerX'], data['playerY'], data['points_non_norm'], data['num_of_points']

        cons = cons.float().cuda(args.gpu)
        playerX = playerX.long().cuda(args.gpu)
        playerY = playerY.long().cuda(args.gpu)
        context = {'pos': cons, 'playerX': playerX, 'playerY': playerY}
        with torch.no_grad():  
            _, samples = model.generate(context, num_points=args.val_sample_points)
            for idx in range(points.size(0)):
                num_of_points = num_points[idx]
                sample = samples[idx].cpu().detach().numpy() * std + mean
                ref = points_all[idx][:num_of_points, :].detach().numpy() * std + mean
                _, jsd = jsd_between_list_of_points(sample, ref, space)
                MSE.append(heatmap_MSE(sample, ref, nn))
                JSD.append(jsd)
            cd, emd = cal_CD_batch(samples, points_all.float().cuda(args.gpu))
            CD.append(cd.cpu().detach().numpy())
            EMD.append(emd.cpu().detach().numpy())

        if bidx == 0:
            results = []
            for idx in range(min(10, points.size(0))):
                if args.no_player_embedding:
                    x = np.argmax(playerX[idx].cpu().detach().numpy())
                    y = np.argmax(playerY[idx].cpu().detach().numpy())
                    cls_num = str(x) + '_' + str(y)
                else:
                    x = playerX[idx].cpu().detach().numpy()[0]
                    y = playerY[idx].cpu().detach().numpy()[0]
                    cls_num = str(x) + '_' + str(y)
                num_of_points = num_points[idx]
                res = visualize_hit_points(samples[idx], points_all[idx][:num_of_points, :], idx, cls_num, m=mean, s=std, space=space)
                results.append(res)
            res = np.concatenate(results, axis=1)
            imageio.imsave(os.path.join(save_dir, 'images_val', tr_val+'_vis_conditioned_epoch%d-gpu%s.png' % (epoch, args.gpu)),
                            res.transpose((1, 2, 0)))
            if writer is not None:
                writer.add_image(tr_val+'_vis/conditioned', torch.as_tensor(res), epoch)
    
    CD = np.concatenate(CD)
    EMD = np.concatenate(EMD)
    writer.add_scalar(tr_val+'/MSE', np.array(MSE).mean(), epoch)
    writer.add_scalar(tr_val+'/JSD', np.array(JSD).mean(), epoch)
    writer.add_scalar(tr_val+'/CD', np.array(CD).mean(), epoch)
    writer.add_scalar(tr_val+'/EMD', np.array(EMD).mean(), epoch)

