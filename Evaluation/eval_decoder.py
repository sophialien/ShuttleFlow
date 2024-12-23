import sys
sys.path.insert(1,'../ShuttleFlow')

import pandas as pd
import numpy as np
import os
import json
import numpy as np

import torch
import torch.nn as nn
import argparse

from datasets import get_datasets
from Baseline.train_decoder import Generator
from Baseline.args import get_parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def metric(model, sample_points, test_data_path, player_list, mean, std):
    batch_size = 64
    test_data = pd.read_csv(test_data_path)

    mae = []
    mse = []
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
                samples = model.sample(context)
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

                    min_square_error = np.round(((samples_stand - ground_truth_stand)**2), 4).sum(axis=2).min(axis=1)
                    min_abs_error = np.round(abs(samples_stand - ground_truth_stand), 4).sum(axis=2).min(axis=1)   
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
    test_data_path = os.path.join('data', dataset, 'test_balls.csv')

    ckpt_name = 'checkpoint-99.pt'
    output_dir = os.path.join("Training_logs", dataset, logname)

    path = os.path.join(output_dir, ckpt_name)
    config_path = os.path.join(output_dir, 'configs.json')

    # player_list = np.load(os.path.join(output_dir, 'player_list.npy'), allow_pickle=True)
    # print(player_list)

    # parser = get_parser()
    # args = parser.parse_args(''.split())

    # with open(config_path) as f:
    #     print('load config from:', config_path)
    #     config = json.load(f)

    # args_dict = vars(args) 
    # args_dict.update(config)
    # print(args)

    # tr_dataset, _, _ = get_datasets(args)
    # player_list = tr_dataset.get_player_list()
    # mean, std = tr_dataset.get_pc_stats()

    # model = Generator(args, tr_dataset.get_num_players())
    # model = model.cuda()
    # ckpt = torch.load(path)
    # model.load_state_dict(ckpt['model'])
    
    # RMSE = []
    # MAE = []
    # RMSE_S = []
    # MAE_S = []
    # for n in [16, 32, 64, 96, 128, 192, 256]:
    #     rmse, mae, rmse_s, mae_s = metric(model, n, test_data_path, player_list, mean, std)
    #     RMSE.append(rmse)
    #     MAE.append(mae)
    #     RMSE_S.append(rmse_s)
    #     MAE_S.append(mae_s)
    # temp = {'rmse': RMSE, 'mae': MAE, 'rmse_s': RMSE_S, 'mae_s': MAE_S}

    # if not os.path.exists('Eval_result/'+dataset):
    #     os.mkdir('Eval_result/'+dataset)
    
    # with open('Eval_result'+dataset+'/'+logname+'.txt', 'w') as json_file:
    #     json.dump(temp, json_file)
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logname', type=str)
    args = parser.parse_args()
    
    dataset = args.dataset
    logname = args.logname

    main(dataset, logname)