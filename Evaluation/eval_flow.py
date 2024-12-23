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
from models.networks import HitPointFlow
from args import get_parser

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def metric(model, generate_points, sample_points, test_data_path, num_past_balls, player_list, mean, std):
    batch_size = 64
    test_data = pd.read_csv(test_data_path)

    batch_data = test_data.iloc[0:2]
    tx, ty, bx, by, x, y = batch_data['pre_x_re'].values, batch_data['pre_y_re'].values, batch_data['x_re'].values, batch_data['y_re'].values, batch_data['next_x_re'].values, batch_data['next_y_re'].values
    cons = np.array([tx, ty, bx, by]).transpose(1, 0)
    
    for k in range(3, num_past_balls+1):
        cons = np.concatenate((batch_data[['pre'+str(k)+'_x_re','pre'+str(k)+'_y_re']].values.astype(float), cons), axis=1)
    print(cons.shape)

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
            
            for k in range(3, num_past_balls+1):
                cons = np.concatenate((batch_data[['pre'+str(k)+'_x_re','pre'+str(k)+'_y_re']].values.astype(float), cons), axis=1)
    
            cons = (cons - np.tile([25, 150], num_past_balls)) / np.tile([300.0, 330.0], num_past_balls) - np.tile([0.5, 0.5], num_past_balls)

            ground_truth = np.array([x, y]).transpose(1, 0)
            ground_truth = ground_truth[:, np.newaxis,:]

            cons = torch.from_numpy(cons).float().cuda()
            playerX = torch.from_numpy(playerX).long().cuda()
            playerY = torch.from_numpy(playerY).long().cuda()
            context = {'pos': cons, 'playerX': playerX, 'playerY': playerY}
            with torch.no_grad():
                _, samples = model.generate(context, num_points=generate_points)
                samples = samples.cpu().detach().numpy()
                
                samples_stand = samples
                ground_truth_stand = (ground_truth - mean) / std

                samples = samples * std + mean

                if sample_points != generate_points:
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
        if sample_points != generate_points:
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

def main(dataset, logname, num_past_balls):
    dataset_path = 'data/' + dataset
    if num_past_balls==2:
        test_data_path = os.path.join(dataset_path, 'test_balls.csv')
    else:
        test_data_path = os.path.join(dataset_path, 'test_balls'+str(num_past_balls)+'.csv')

    ckpt_name = 'checkpoint-99.pt'
    output_dir = os.path.join("Training_logs", dataset, logname)
    
    path = os.path.join(output_dir, ckpt_name)
    config_path = os.path.join(output_dir, 'configs.json')

    player_list = np.load(os.path.join(output_dir, 'player_list.npy'), allow_pickle=True)
    print(player_list)

    parser = get_parser()
    args = parser.parse_args(''.split())

    with open(config_path) as f:
        print('load config from:', config_path)
        config = json.load(f)

    args_dict = vars(args) 
    args_dict.update(config)
    print(args)

    tr_dataset, _, _ = get_datasets(args)

    player_list = tr_dataset.get_player_list()
    mean, std = tr_dataset.get_pc_stats()

    model = HitPointFlow(args, tr_dataset.get_num_players())
    def _transform_(m):
        return nn.DataParallel(m)
    model = model.cuda()
    model.multi_gpu_wrapper(_transform_)

    ckpt = torch.load(path)
    model.load_state_dict(ckpt['model'])

    print(ckpt_name)
    print("Number of trainable parameters of model: {}".format(count_parameters(model)))

    RMSE = []
    MAE = []
    RMSE_S = []
    MAE_S = []
    for n in [16, 32, 64, 96, 128, 192, 256]:
        rmse, mae, rmse_s, mae_s = metric(model, 256, n, test_data_path, num_past_balls, player_list, mean, std)
        RMSE.append(rmse)
        MAE.append(mae)
        RMSE_S.append(rmse_s)
        MAE_S.append(mae_s)
    temp = {'rmse': RMSE, 'mae': MAE, 'rmse_s': RMSE_S, 'mae_s': MAE_S}
  
    if not os.path.exists('Eval_result/'+dataset):
        os.mkdir('Eval_result/'+dataset)

    with open('Eval_result/'+dataset+'/'+logname+'.txt', 'w') as json_file:
        json.dump(temp, json_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--logname', type=str)
    parser.add_argument('--num_past_balls', type=int, default=2)
    args = parser.parse_args()
    
    dataset = args.dataset
    logname = args.logname
    num_past_balls = args.num_past_balls

    main(dataset, logname, num_past_balls)