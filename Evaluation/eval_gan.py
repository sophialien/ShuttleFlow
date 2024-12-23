import sys
sys.path.insert(1,'/mnt/train-data1/dandan/HitPointFlow')

from GAN_models.gan import Discriminator, Generator, Context_encoder
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import argparse

import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"]= '2'

from datasets import get_datasets
from args_gan import get_parser
from utils import heatmap_MSE, visualize_hit_points, draw_heatmap
from sklearn.neighbors import NearestNeighbors

from scipy import ndimage


def metric1(Decoder, sample_points, test_data_path, player_list, z2_dim, mean, std):
    batch_size = 64
    test_data = pd.read_csv(test_data_path)

    Decoder.eval()

    mse = []
    mae = []
    mse_stand = []
    mae_stand = []
    for i in range(5):
        sq_error = []
        abs_error = []
        sq_error_stand = []
        abs_error_stand = []
        for start in range(0, len(test_data), batch_size):
            batch_data = test_data.iloc[start:start+batch_size]
            playerX, playerY, tx, ty, bx, by, x, y = batch_data['playerX'].values, batch_data['playerY'].values, batch_data['pre_x_re'].values, batch_data['pre_y_re'].values, batch_data['x_re'].values, batch_data['y_re'].values, batch_data['next_x_re'].values, batch_data['next_y_re'].values
            playerX = np.array(list(map(lambda player: np.where(player_list==player.replace(' ', '_'))[0], playerX)))
            playerY = np.array(list(map(lambda player: np.where(player_list==player.replace(' ', '_'))[0], playerY)))
            cons = np.array([tx, ty, bx, by]).transpose(1, 0)
            cons = (cons - [25, 150, 25, 150]) / [300.0, 330.0, 300.0, 330.0] - [0.5, 0.5, 0.5, 0.5]

            ground_truth = np.array([x, y]).transpose(1, 0)
            ground_truth = ground_truth[:, np.newaxis,:]

            cons = torch.from_numpy(cons).float().cuda()
            playerX = torch.from_numpy(playerX).long().cuda()
            playerY = torch.from_numpy(playerY).long().cuda()
            context = {'pos': cons, 'playerX': playerX, 'playerY': playerY}
            with torch.no_grad():
                z2 = torch.randn(batch_data.shape[0], 256, z2_dim).cuda()
                samples = Decoder(context, z2)
                samples = samples.cpu().detach().numpy()
                
                samples_stand = samples
                ground_truth_stand = (ground_truth - mean) / std

                samples = samples * std + mean

                if sample_points != 256:
                    for idx in range(len(samples)):
                        sampled_idx = np.random.choice(256, sample_points)
                        predicted = samples[idx, sampled_idx, :]
                        predicted_stand = samples_stand[idx, sampled_idx, :]
                        min_square_error = ((predicted - ground_truth[idx])**2).sum(axis=1).min()
                        min_abs_error = abs(predicted - ground_truth[idx]).sum(axis=1).min()
                        sq_error.append(min_square_error)
                        abs_error.append(min_abs_error)
                        
                        min_square_error = np.round(((predicted_stand - ground_truth_stand[idx])**2), 4).sum(axis=1).min()
                        min_abs_error = np.round(abs(predicted_stand - ground_truth_stand[idx]), 4).sum(axis=1).min()   
                        sq_error_stand.append(min_square_error)
                        abs_error_stand.append(min_abs_error)
                else:
                    min_square_error = ((samples - ground_truth)**2).sum(axis=2).min(axis=1)
                    min_abs_error = abs(samples - ground_truth).sum(axis=2).min(axis=1)   
                    sq_error.append(min_square_error)
                    abs_error.append(min_abs_error)

                    min_square_error = ((samples_stand - ground_truth_stand)**2).sum(axis=2).min(axis=1)
                    min_abs_error = abs(samples_stand - ground_truth_stand).sum(axis=2).min(axis=1)   
                    sq_error_stand.append(min_square_error)
                    abs_error_stand.append(min_abs_error)
        if sample_points != 256:
            sq_error = np.array(sq_error)
            abs_error = np.array(abs_error)
            sq_error_stand = np.array(sq_error_stand)
            abs_error_stand = np.array(abs_error_stand)
        else:
            sq_error = np.concatenate(sq_error)
            abs_error = np.concatenate(abs_error)
            sq_error_stand = np.concatenate(sq_error_stand)
            abs_error_stand = np.concatenate(abs_error_stand)
        mse.append(sq_error.mean())
        mae.append(abs_error.mean())
        mse_stand.append(sq_error_stand.mean())
        mae_stand.append(abs_error_stand.mean())
    print("sample points=", sample_points)
    print("MSE: ", np.array(mse).mean(), "MAE: ", np.array(mae).mean())
    print(mse, mae)
    print("MSE standarized: ", np.array(mse_stand).mean(), "MAE: ", np.array(mae_stand).mean())
    print(mse_stand, mae_stand)

    return np.sqrt(np.array(mse).mean()), np.array(mae).mean(), np.sqrt(np.array(mse_stand).mean()), np.array(mae_stand).mean()

def main(dataset, logname):
    test_folder = 'data_newM'
    sub_folder = 'flow_' + dataset + '_grid'
    test_data_path = os.path.join(test_folder, sub_folder, 'test_balls.csv')

    ckpt_name = 'checkpoint-29.pt'
    output_dir = os.path.join("Training_logs", dataset, logname)

    path = os.path.join(output_dir, ckpt_name)
    config_path = os.path.join(output_dir, 'configs.json')

    parser = get_parser()
    args_g = parser.parse_args(''.split())

    with open(config_path) as f:
        print('load config from:', config_path)
        config_g = json.load(f)

    args_dict_g = vars(args_g) 
    args_dict_g.update(config_g)

    space = {'w_space': args_g.w_space, 'h_space': args_g.h_space}

    tr_dataset, te_dataset, _ = get_datasets(args_g)
    test_loader = torch.utils.data.DataLoader(
            dataset=te_dataset, batch_size=args_g.batch_size, shuffle=True,
            pin_memory=True, drop_last=False)

    player_list = tr_dataset.get_player_list()
    mean, std = tr_dataset.get_pc_stats()

    Decoder = Generator(args_g, tr_dataset.get_num_players()).cuda()

    ckpt_g = torch.load(path)
    Decoder.load_state_dict(ckpt_g['generator'])

    RMSE = []
    MAE = []
    RMSE_S = []
    MAE_S = []
    for n in [16, 32, 64, 96, 128, 192, 256]:
    # for n in range(10, 260, 10):
    # for n in [2, 4, 8, 16, 32, 64, 128, 256]:
    # for n in [10, 32, 64, 96, 128, 192, 256]:
        rmse, mae, rmse_s, mae_s = metric1(Decoder, n, test_data_path, player_list, args_g.z2_dim, mean, std)
        RMSE.append(rmse)
        MAE.append(mae)
        RMSE_S.append(rmse_s)
        MAE_S.append(mae_s)
    temp = {'rmse': RMSE, 'mae': MAE, 'rmse_s': RMSE_S, 'mae_s': MAE_S}

    if not os.path.exists('Eval_result_E30/'+sub_folder):
        os.mkdir('Eval_result_E30/'+sub_folder)
    
    with open('Eval_result_E30/'+sub_folder+'/'+logname+'.txt', 'w') as json_file:
        json.dump(temp, json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logname', type=str)
    args = parser.parse_args()
    
    dataset = args.dataset
    logname = args.logname

    main(dataset, logname)

