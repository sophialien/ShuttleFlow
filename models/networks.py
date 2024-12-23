import torch
import numpy as np
import torch.nn.functional as F
from torch import optim
from torch import nn
from metrics.CD_EMD import cal_CD, cal_CD_batch
from models.flow import get_hit_point_cnf
from models.position_embedder import get_embedder
from utils import truncated_normal, reduce_tensor, standard_normal_logprob

# Model
class HitPointFlow(nn.Module):
    def __init__(self, args, num_player, is_latent=False):
        super(HitPointFlow, self).__init__()
        self.input_dim = args.input_dim if not is_latent else args.z1_dim
        self.distributed = args.distributed
        self.truncate_std = None

        self.context_cat = args.context_cat
        self.pair_condition = args.pair_condition
        
        if (self.context_cat != 'concat') or (args.condition_type == 'discrete'):
            assert (args.num_past_balls == 2), "num_past_balls > 2 do not support 'add' context operation"
        position_dim = args.num_past_balls * 2 if self.context_cat == 'concat' else 2
        
        if args.no_pos_embedding:
            self.embed_fn, self.zdim = get_embedder(args.multires, position_dim,  -1)
        else:
            self.embed_fn, self.zdim = get_embedder(args.multires, position_dim, 0)

        if self.pair_condition:
            if not args.no_player_embedding:
                if self.context_cat == 'concat':
                    self.player_embedder = nn.Embedding(num_player, args.player_style_dim)
                    self.pdim = args.player_style_dim * 2
                    self.zdim = self.zdim + self.pdim
                else:
                    self.player_embedder = nn.Embedding(num_player, self.zdim)
                    self.pdim = None
                    self.zdim = self.zdim * 2
            else:
                self.player_embedder = nn.Identity()
                self.pdim = num_player * 2
                self.zdim = self.zdim + self.pdim
            
            # self.layers = nn.Sequential(
            #     nn.Linear(self.zdim, 128),
            #     nn.Softplus(),
            #     nn.Linear(128, 128),
            # )
            # self.zdim = 128
        else:
            self.pdim = None
        
        if is_latent:
            self.hit_point_cnf = get_hit_point_cnf(args, self.input_dim, self.zdim, self.pdim)
        else:
            self.hit_point_cnf = get_hit_point_cnf(args, self.input_dim, self.zdim, self.pdim)

        print('z-dim: ', self.zdim)

    @staticmethod
    def sample_gaussian(size, truncate_std=None, gpu=None):
        y = torch.randn(*size).float()
        y = y if gpu is None else y.cuda(gpu)
        if truncate_std is not None:
            truncated_normal(y, mean=0, std=1, trunc_std=truncate_std)
        return y

    @staticmethod
    def reparameterize_gaussian(mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(std.size()).to(mean)
        return mean + std * eps

    def multi_gpu_wrapper(self, f):
        self.hit_point_cnf = f(self.hit_point_cnf)

    def make_optimizer(self, args):
        def _get_opt_(params):
            if args.optimizer == 'adam':
                optimizer = optim.Adam(params, lr=args.lr, betas=(args.beta1, args.beta2),
                                       weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
            else:
                assert 0, "args.optimizer should be either 'adam' or 'sgd'"
            return optimizer
        opt = _get_opt_(list(self.hit_point_cnf.parameters()))
        return opt

    def forward(self, x, context, opt, step, writer=None):
        # opt.zero_grad()
        batch_size = x.size(0)
        num_points = x.size(1)

        # Compute the reconstruction likelihood P(X|z)
        if self.pair_condition:
            if self.context_cat == 'concat':
                context['pos'] = self.embed_fn(context['pos'])
                player_embedding = torch.cat((self.player_embedder(context['playerX']), self.player_embedder(context['playerY'])), dim=-1)
                player_embedding = torch.squeeze(player_embedding, 1)
                context = torch.cat((context['pos'], player_embedding), dim=1)
                # context = self.layers(context)
            else:
                bot_context = self.embed_fn(context['pos'][:, 2:]) + self.player_embedder(context['playerX']).squeeze(1)
                top_context = self.embed_fn(context['pos'][:, :2]) + self.player_embedder(context['playerY']).squeeze(1)
                context = torch.cat((top_context, bot_context), dim=1)
                
        else:
            context = self.embed_fn(context['pos'])
        # context = self.layers(context).unsqueeze(1)
        y, delta_log_py = self.hit_point_cnf(x, context, torch.zeros(batch_size, num_points, 1).to(x))
        log_py = standard_normal_logprob(y).view(batch_size, -1).sum(1, keepdim=True)
        delta_log_py = delta_log_py.view(batch_size, num_points, 1).sum(1)
        log_px = log_py - delta_log_py

        # Loss
        loss = -log_px.mean()

        # LOGGING (after the training)
        if self.distributed:
            nll = reduce_tensor(-log_px.mean())
        else:
            nll = -log_px.mean()

        bits_per_dim = nll / float(x.size(1) * x.size(2))

        return {
            'bits_per_dim': bits_per_dim,
            'loss': loss,
        }

    def generate(self, context, num_points, temperature=1.0, truncate_std=None):
        # transform points from the prior to a point cloud, conditioned on a shape code
        if self.pair_condition:
            if self.context_cat == 'concat':
                context['pos'] = self.embed_fn(context['pos'])
                player_embedding = torch.cat((self.player_embedder(context['playerX']), self.player_embedder(context['playerY'])), dim=-1)
                player_embedding = torch.squeeze(player_embedding, 1)
                context = torch.cat((context['pos'], player_embedding), dim=1)
                # context = self.layers(context)
            else:
                bot_context = self.embed_fn(context['pos'][:, 2:]) + self.player_embedder(context['playerX']).squeeze(1)
                top_context = self.embed_fn(context['pos'][:, :2]) + self.player_embedder(context['playerY']).squeeze(1)
                context = torch.cat((top_context, bot_context), dim=1)
        else:
            context = self.embed_fn(context['pos'])
        # context = self.layers(context).unsqueeze(1)
        y = self.sample_gaussian((context.size(0), num_points, self.input_dim), truncate_std).to(context) * temperature
        x = self.hit_point_cnf(y, context, reverse=True).view(*y.size())
        return y, x

    def regularization(self, context_A, context_B, points_A, points_B, num_points, criterion, step, writer):
        playerA = context_A['playerX']
        if self.pair_condition:
            if self.context_cat == 'concat':
                context_A['pos'] = self.embed_fn(context_A['pos'])
                player_embedding = torch.cat((self.player_embedder(context_A['playerX']), self.player_embedder(context_A['playerY'])), dim=-1)
                player_embedding = torch.squeeze(player_embedding, 1)
                context_A = torch.cat((context_A['pos'], player_embedding), dim=1)
                context_B['pos'] = self.embed_fn(context_B['pos'])
                player_embedding = torch.cat((self.player_embedder(context_B['playerX']), self.player_embedder(context_B['playerY'])), dim=-1)
                player_embedding = torch.squeeze(player_embedding, 1)
                context_B = torch.cat((context_B['pos'], player_embedding), dim=1)
            else:
                bot_context = self.embed_fn(context_A['pos'][:, 2:]) + self.player_embedder(context_A['playerX']).squeeze(1)
                top_context = self.embed_fn(context_A['pos'][:, :2]) + self.player_embedder(context_A['playerY']).squeeze(1)
                context_A = torch.cat((top_context, bot_context), dim=1)
                bot_context = self.embed_fn(context_B['pos'][:, 2:]) + self.player_embedder(context_B['playerX']).squeeze(1)
                top_context = self.embed_fn(context_B['pos'][:, :2]) + self.player_embedder(context_B['playerY']).squeeze(1)
                context_B = torch.cat((top_context, bot_context), dim=1)
        else:
            context_A = self.embed_fn(context_A['pos'])
            context_B = self.embed_fn(context_B['pos'])
        # context_A = self.layers(context_A).unsqueeze(1)
        # context_B = self.layers(context_B).unsqueeze(1)
        y = self.sample_gaussian((context_A.size(0), num_points, self.input_dim))
        x_AC = self.hit_point_cnf(y, context_A, reverse=True).view(*y.size())
        x_BC = self.hit_point_cnf(y, context_B, reverse=True).view(*y.size())
        
        cd_r, emd_r = cal_CD_batch(points_A, points_B)
        cd_l, emd_l = cal_CD_batch(x_AC, x_BC)

        mae = criterion(emd_l, emd_r, playerA)

        return mae

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.mae = nn.L1Loss()

    def forward(self, emd_l, emd_r, player):
        return self.mae(emd_l, self.alpha * emd_r)

class MAELoss_alphas(nn.Module):
    def __init__(self, num_player):
        super(MAELoss_alphas, self).__init__()
        self.alpha = nn.Embedding(num_player, 1)
        self.mae = nn.L1Loss()

    def forward(self, emd_l, emd_r, player):
        a = self.alpha(player)
        return self.mae(emd_l, a * emd_r)