import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import warnings
import torch.distributed
import numpy as np
import random
import faulthandler
import torch.multiprocessing as mp
import time
import imageio
from models.networks import HitPointFlow, MAELoss, MAELoss_alphas
from torch import optim
from args import get_args
from torch.backends import cudnn
from utils import AverageValueMeter, set_random_seed, save, resume, visualize_hit_points
from tensorboardX import SummaryWriter
from datasets import get_datasets, init_np_seed
from utils import validate

import json

faulthandler.enable()


def main_worker(gpu, save_dir, ngpus_per_node, args):
    # basic setup
    cudnn.benchmark = True
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.log_name is not None:
        log_dir = "Training_logs/%s" % args.log_name
    else:
        log_dir = "Training_logs/time-%d" % time.time()

    if not args.distributed or (args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(logdir=log_dir)
    else:
        writer = None

    # initialize datasets and loaders
    tr_dataset, te_dataset, re_dataset = get_datasets(args)
    mean, std = tr_dataset.get_pc_stats()
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(tr_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        dataset=tr_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler, drop_last=True,
        worker_init_fn=init_np_seed)
    test_loader = torch.utils.data.DataLoader(
        dataset=te_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False,
        worker_init_fn=init_np_seed)
    if re_dataset is not None:
        reg_loader = torch.utils.data.DataLoader(
            dataset=re_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
            worker_init_fn=init_np_seed)

        reg_criterion = MAELoss().cuda(args.gpu)
        # reg_criterion = MAELoss_alphas(tr_dataset.get_num_players()).cuda(args.gpu)
        # optimizer_criterion = optim.Adam(reg_criterion.parameters(), lr=args.lr)

    np.save(os.path.join(save_dir, "player_list.npy"), tr_dataset.get_player_list())
    # save dataset statistics
    if not args.distributed or (args.rank % ngpus_per_node == 0):
        np.save(os.path.join(save_dir, "train_set_mean.npy"), tr_dataset.all_points_mean)
        np.save(os.path.join(save_dir, "train_set_std.npy"), tr_dataset.all_points_std)
        np.save(os.path.join(save_dir, "val_set_mean.npy"), te_dataset.all_points_mean)
        np.save(os.path.join(save_dir, "val_set_std.npy"), te_dataset.all_points_std)

    # multi-GPU setup
    model = HitPointFlow(args, tr_dataset.get_num_players())
    if args.distributed:  # Multiple processes, single GPU per process
        if args.gpu is not None:
            def _transform_(m):
                return nn.parallel.DistributedDataParallel(
                    m, device_ids=[args.gpu], output_device=args.gpu, check_reduction=True)

            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model.multi_gpu_wrapper(_transform_)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = 0
        else:
            assert 0, "DistributedDataParallel constructor should always set the single device scope"
    elif args.gpu is not None:  # Single process, single GPU per process
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:  # Single process, multiple GPUs per process
        def _transform_(m):
            return nn.DataParallel(m)
        model = model.cuda()
        model.multi_gpu_wrapper(_transform_)

    # resume checkpoints
    start_epoch = 0
    optimizer = model.make_optimizer(args)
    optimizer_reg = optim.Adam(model.parameters(), lr=args.lr)
    if args.resume_checkpoint is None and os.path.exists(os.path.join(save_dir, 'checkpoint-latest.pt')):
        args.resume_checkpoint = os.path.join(save_dir, 'checkpoint-latest.pt')  # use the latest checkpoint
    if args.resume_checkpoint is not None:
        if args.resume_optimizer:
            model, optimizer, start_epoch = resume(
                args.resume_checkpoint, model, optimizer, strict=(not args.resume_non_strict))
        else:
            model, _, start_epoch = resume(
                args.resume_checkpoint, model, optimizer=None, strict=(not args.resume_non_strict))
        print('Resumed from: ' + args.resume_checkpoint)

    # initialize the learning rate scheduler
    if args.scheduler == 'exponential':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.exp_decay)
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 2, gamma=0.1)
    elif args.scheduler == 'linear':
        def lambda_rule(ep):
            lr_l = 1.0 - max(0, ep - 0.5 * args.epochs) / float(0.5 * args.epochs)
            return lr_l
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    else:
        assert 0, "args.schedulers should be either 'exponential' or 'linear'"

    # main training loop
    start_time = time.time()
    bits_per_dim_avg_meter = AverageValueMeter()
    if args.distributed:
        print("[Rank %d] World size : %d" % (args.rank, dist.get_world_size()))

    print("Start epoch: %d End epoch: %d" % (start_epoch, args.epochs))
    
    step = 0
    # validate(test_loader, model, 0, writer, save_dir, args, mean, std, 'val')
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # adjust the learning rate
        if (epoch !=0) and (epoch + 1) % args.exp_decay_freq == 0:
            scheduler.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar('lr/optimizer', scheduler.get_lr()[0], epoch)

        # train for one epoch
        for bidx, data in enumerate(train_loader):
            idx_batch, tr_points, tr_cons, tr_playerX, tr_playerY, tr_points_all, tr_num_points = \
                        data['idx'], data['points'], data['cons'], data['playerX'], data['playerY'], data['points_non_norm'], data['num_of_points']
            step += 1
            model.train()

            optimizer.zero_grad()

            input_points = tr_points.float().cuda(args.gpu, non_blocking=True)
            input_cons = tr_cons.float().cuda(args.gpu, non_blocking=True)
            input_playerX = tr_playerX.long().cuda(args.gpu, non_blocking=True)
            input_playerY = tr_playerY.long().cuda(args.gpu, non_blocking=True)
            context = {'pos': input_cons, 'playerX': input_playerX, 'playerY': input_playerY}

            out = model(input_points, context, optimizer, step, writer)
            nll = out['loss']
            bits_per_dim = out['bits_per_dim']
            bits_per_dim_avg_meter.update(bits_per_dim)

            nll.backward()

            if re_dataset is not None:
                reg_it = iter(reg_loader)
                data = next(reg_it)
                idx_batch, playerA, playerB, playerC, cons, points_A, points_B = \
                        data['idx'], data['playerA'], data['playerB'], data['playerC'], \
                            data['cons'], data['points_A'], data['points_B']
                points_A = points_A.float().cuda(args.gpu, non_blocking=True)
                points_B = points_B.float().cuda(args.gpu, non_blocking=True)
                cons = cons.float().cuda(args.gpu, non_blocking=True)
                playerA = playerA.long().cuda(args.gpu, non_blocking=True)
                playerB = playerB.long().cuda(args.gpu, non_blocking=True)
                playerC = playerC.long().cuda(args.gpu, non_blocking=True)
                context_A = {'pos': cons, 'playerX': playerA, 'playerY': playerC}
                context_B = {'pos': cons, 'playerX': playerB, 'playerY': playerC}

                if args.regularization_type == 'nll':
                    out = model(points_A, context_A, optimizer, step, writer)
                    reg_loss = out['loss']
                    bits_per_dim_reg = out['bits_per_dim']
                elif args.regularization_type == 'mae':
                    reg_loss = model.regularization(context_A, context_B, points_A, points_B, args.sample_points, reg_criterion, step, writer)
                    
                (reg_loss * args.regularization_w).backward()
                loss = nll + reg_loss * args.regularization_w
            else:
                loss = nll

            optimizer.step()

            if writer is not None:
                writer.add_scalar('train/total_loss', loss, step)
                writer.add_scalar('train/nll', nll, step)
                writer.add_scalar('train/bits_per_dim', bits_per_dim, step)
                if re_dataset is not None:
                    writer.add_scalar('train/reg', reg_loss, step)

            if step % args.log_freq == 0:
                duration = time.time() - start_time
                start_time = time.time()
                print("[Rank %d] Epoch %d Batch [%2d/%2d] Time [%3.2fs] Bits-per-dim %2.5f"
                      % (args.rank, epoch, bidx, len(train_loader), duration, bits_per_dim_avg_meter.avg))
      
        # evaluate on the validation set
        if not args.no_validation and (epoch + 1) % args.val_freq == 0:
            pass
            # validate(test_loader, model, step, writer, save_dir, args, mean, std, 'val')
            
        # save visualizations
        if (epoch + 1) % args.viz_freq == 0:
            model.eval()

            tr_cons = tr_cons.float().cuda(args.gpu)
            tr_playerX = tr_playerX.long().cuda(args.gpu)
            tr_playerY = tr_playerY.long().cuda(args.gpu)
            context = {'pos': tr_cons, 'playerX': tr_playerX, 'playerY': tr_playerY}
            with torch.no_grad():
                _, samples = model.generate(context, num_points=args.val_sample_points)
            results = []

            mean, std = tr_dataset.get_pc_stats()
            for idx in range(min(10, tr_points.size(0))):
                if args.no_player_embedding:
                    x = np.argmax(tr_playerX[idx].cpu().detach().numpy())
                    y = np.argmax(tr_playerY[idx].cpu().detach().numpy())
                    cls_num = str(x) + '_' + str(y)
                else:
                    x = tr_playerX[idx].cpu().detach().numpy()[0]
                    y = tr_playerY[idx].cpu().detach().numpy()[0]
                    cls_num = str(x) + '_' + str(y)
                
                num_of_points = tr_num_points[idx]
                space = {'w_space': args.w_space, 'h_space': args.h_space}
                res = visualize_hit_points(samples[idx], tr_points_all[idx][:num_of_points, :], idx, cls_num, m=mean, s=std, space=space)
                results.append(res)
            res = np.concatenate(results, axis=1)
            imageio.imsave(os.path.join(save_dir, 'images', 'tr_vis_conditioned_epoch%d-gpu%s.png' % (epoch, args.gpu)),
                              res.transpose((1, 2, 0)))
            if writer is not None:
                writer.add_image('train_vis/conditioned', torch.as_tensor(res), epoch)
        

        # save checkpoints
        if not args.distributed or (args.rank % ngpus_per_node == 0):
            if (epoch == 0) or ((epoch + 1) % args.save_freq == 0):
                save(model, optimizer, epoch + 1,
                     os.path.join(save_dir, 'checkpoint-%d.pt' % epoch))
                save(model, optimizer, epoch + 1,
                     os.path.join(save_dir, 'checkpoint-latest.pt'))
        

def main():
    # command line args
    args = get_args()
    save_dir = os.path.join('Training_logs', args.log_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'images'))
        os.makedirs(os.path.join(save_dir, 'images_val'))

    with open(os.path.join(save_dir, 'command.sh'), 'w') as f:
        f.write('python -X faulthandler ' + ' '.join(sys.argv))
        f.write('\n')

    with open(os.path.join(save_dir, 'configs.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    if args.seed is None:
        args.seed = random.randint(0, 1000000)
    set_random_seed(args.seed)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.sync_bn:
        assert args.distributed

    print("Arguments:")
    print(args)

    ngpus_per_node = torch.cuda.device_count()
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(save_dir, ngpus_per_node, args))
    else:
        main_worker(args.gpu, save_dir, ngpus_per_node, args)


if __name__ == '__main__':
    main()
