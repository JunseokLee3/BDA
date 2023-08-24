import wandb
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from BDA_model import BDA
from dataset import AWA2DataLoader
from  helper_func import eval_zs_gzsl
import os

# init wandb from config file
if not os.path.isdir('checkpoint'):
    os.makedirs('checkpoint')

# load dataset
dataloader = AWA2DataLoader('.', 'cuda:0')

sweep_configuration = {
    'method': 'grid',
    'name': 'AwA2_CZSL',
    'metric': {'goal': 'maximize', 'name': 'best_H'},
    'parameters': 
    {   'batch_size':    {'value': 50},
        'dataset': { 'value': 'AWA2'},
        'device': {'value': 'cuda:0'},
        'dim_f': { 'value': 2048},
        'dim_v':{ 'value': 300},
        'epochs': {'value': 200},
        'img_size': {'value': 448},
        'lambda_': {'values': [2.5]},
        'lambda_reg': {'values': [0.0001]},
        'lambda_CEVi': {'values': [0.0025]},
        'normalize_V': {'value': "False"},
        'num_attribute': { 'value': 85},
        'num_class': {'value': 50},
        'random_seed': {'value': 17},
        'tf_SAtt':  {'value':  "true"},
        'tf_aux_embed': {'value': "True"},
        'tf_common_dim': {'values': [500]},  
        'tf_dc_layer': {'value': 1},
        'tf_dim_feedforward': {'values': [2048]},
        'tf_dropout': {'values':[0.45]},
        'tf_dropout1': {'values': [0.45]},
        'dropout_self' : {'values': [0]},

        'tf_ec_layer': {'value':1},
        'tf_heads': {'value': 1},
        'alpha': {'value': 1},
        'beta' : {'value': 1},
        'fsqrt': {'values': [127]},
        'sqrt': {'values': [ 153]},        
        'weight' : {'values': [0.01]} ,
        'weight_self' : {'values': [1]},
        'tgt_weight' : {'values': [1]}                         
        }
                                                }
sweep_id = wandb.sweep(sweep=sweep_configuration, project='bda')

def main():
    run = wandb.init()
    config = wandb.config
    print('Config file from wandb:', config)     
    # set random seed
    seed = config.random_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    init_w2v_att = dataloader.w2v_att
    att = dataloader.att
    att[att<0] = 0
    normalize_att = dataloader.normalize_att

     #  model
    model = BDA(config, att, init_w2v_att,
                    dataloader.seenclasses, dataloader.unseenclasses).to(config.device)
    optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0001, momentum=0.)

    # main loop
    niters = dataloader.ntrain * config.epochs//config.batch_size
    report_interval = niters//config.epochs
    best_performance = [0, 0, 0, 0]
    best_performance_zsl = 0
    for i in range(0, niters):
        model.train()
        optimizer.zero_grad()
        
        batch_label, batch_feature, batch_att = dataloader.next_batch(config.batch_size)
        out_package = model(batch_feature)
        
        in_package = out_package
        in_package['batch_label'] = batch_label
        
        out_package = model.compute_loss(in_package)
        loss, loss_CE, loss_cal, loss_reg, loss_CE_Ve = out_package['loss'], out_package[
            'loss_CE'], out_package['loss_cal'], out_package['loss_reg'], out_package['loss_CE_Ve']

        loss.backward()
        optimizer.step()

        # report result
        if i % report_interval==0:
            print('-'*30)
            acc_seen, acc_novel, H, acc_zs = eval_zs_gzsl(
                dataloader, model, config.device, bias_seen=0, bias_unseen=0)
            
            if H > best_performance[2]:
                best_performance = [acc_novel, acc_seen, H, acc_zs]
               
          
            if acc_zs > best_performance_zsl:
                best_performance_zsl = acc_zs
                data = {}
                data["model_state_dict"] = model.state_dict()
                data["optimizer_state_dict"] = optimizer.state_dict(),
                data['epoch'] = int(i//report_interval)
            
                model_file_path = './checkpoint/AwA2_CZSL.pth'
                torch.save(data, model_file_path)
                print('save model: ' + model_file_path)                  
                  

            print('iter/epoch=%d/%d | loss=%.3f, loss_CE=%.3f, loss_cal=%.3f, '
                'loss_reg=%.3f loss_CE_Ve=%.3f | acc_unseen=%.3f, acc_seen=%.3f, H=%.3f | '
                'acc_zs=%.3f' % (
                    i, int(i//report_interval),
                    loss.item(), loss_CE.item(), loss_cal.item(),
                    loss_reg.item(), loss_CE_Ve.item(),
                    best_performance[0], best_performance[1],
                    best_performance[2], best_performance_zsl))

            wandb.log({
                'iter': i,
                'loss': loss.item(),
                'loss_CE': loss_CE.item(),
                'loss_cal': loss_cal.item(),
                'loss_reg': loss_reg.item(),
                'loss_CE_Ve': loss_CE_Ve.item(),
                'acc_unseen': acc_novel,
                'acc_seen': acc_seen,
                'H': H,
                'acc_zs': acc_zs,
                'best_acc_unseen': best_performance[0],
                'best_acc_seen': best_performance[1],
                'best_H': best_performance[2],
                'best_acc_zs': best_performance_zsl
            })

wandb.agent(sweep_id, function=main, count=1 )
