# Train.py
#     - model training
#     - selects cliend

import argparse
import random
from datetime import datetime
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from server import Server
from client import Client
from datasets import MIDataset, EA, ArrayToTensor
from utils import EarlyStopping
from attack import HonestButCuriousServer


def strtobool(val):
    return str(val).lower() in ("yes", "true", "t", "1")

def train(args, server_subject_id, client_subject_id, Server_TestAcc_List, trace_func=print, save_path='./checkpoint.pth', Attack_Success_List=None):
    seed = random.randint(1, 100) 
    data_transform = [
        EA() if args.ea else None,
        ArrayToTensor()
    ]
    label_transform = [ArrayToTensor()]

    # 1. Initialize Server Dataset
    test_dataset = MIDataset(random_state=seed, subject_id=server_subject_id, root=args.data_path,
                             mode='all', data_transform=data_transform, label_transform=label_transform)
        
    early_stopping = EarlyStopping(patience=args.patience, verbose=False, delta=0, 
                                   path=save_path, trace_func=trace_func, counter_info=False, is_save=True, early=args.early)

    trace_func(f'Begin Initing Server {server_subject_id} & Clients {client_subject_id}')
    server = Server(args, test_dataset)
    
    # 2. Initialize Clients
    clients = []
    for i in client_subject_id:
        clients.append(Client(
            args,
            MIDataset(random_state=seed, subject_id=[i], root=args.data_path, mode='all', data_transform=data_transform, label_transform=label_transform),
            MIDataset(random_state=seed, subject_id=[i], root=args.data_path, mode='all', data_transform=data_transform, label_transform=label_transform), 
            id=i
        ))

    adversary = HonestButCuriousServer()

    trace_func('Begin Training')
    
    # 3. Global Training Loop
    for epoch in range(args.global_epochs):
        candidates = random.sample(clients, args.sample_num)
        candidates_id_list = [j.client_id for j in candidates]
        
        if epoch == 0: 
            avg_weight_dict = {id: 1/len(candidates_id_list) for id in client_subject_id}

        eval_loss = 0
        eval_acc = 0
        client_weight_dict = {} 
        
        for j in candidates:
            client_weight_dict[j.client_id] = {name: torch.zeros_like(params) for name, params in server.global_model.state_dict().items()}

        for j in candidates:
            # Local Training
            weight_diff = j.local_train(server.global_model)

            # --- SNOOP ON THE UPDATE ---
            adversary.snoop_update(weight_diff, client_id=j.client_id)
                
            # Local Evaluation
            loss, acc = j.local_eval(server.global_model)
            eval_loss += loss
            eval_acc += acc
            
            # Accumulate Weights
            for name, params in server.global_model.state_dict().items():
                client_weight_dict[j.client_id][name].add_(weight_diff[name]) 
        
        # Average metrics
        eval_loss /= len(candidates)
        eval_acc /= len(candidates)
        
        early_stopping(eval_loss, server.global_model, eval_acc, epoch, args.global_epochs)

        # Aggregate Weights at Server
        server.model_aggregate(client_weight_dict, avg_weight_dict=avg_weight_dict, candidates_id_list=candidates_id_list)

        if early_stopping.early_stop:
            trace_func(f'Stopped early, stop epoch: {early_stopping.best_epoch+1}')
            trace_func(f'Global Model Val Acc: {100*early_stopping.best_val_acc:.2f}%')
            break
    
    if not early_stopping.early_stop:
        trace_func(f'Not stopped early, stop epoch: {early_stopping.best_epoch+1}')
        trace_func(f'Global Model Val Acc: {100*early_stopping.best_val_acc:.2f}%')

    # 4. Final Server Test
    server.global_model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss, test_acc = server.model_test()
    Server_TestAcc_List.append(round(100*test_acc, 2))
    trace_func(f'Server Test Acc: {100*test_acc:.2f}%')

    # --- EXECUTE THE PRIVACY AUDIT ---
    attack_success = adversary.perform_reidentification_attack()
    trace_func(f'🚨 ATTACK SUCCESS RATE: {attack_success:.2f}%')
    if Attack_Success_List is not None:
        Attack_Success_List.append(round(attack_success, 2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated Learning for BCIs')

    parser.add_argument('--dp', type=lambda x: bool(strtobool(x)), default=False, help='Apply Differential Privacy')
    parser.add_argument('--clip_bound', type=float, default=1.0, help='Max L2 norm for DP clipping')
    parser.add_argument('--noise_multiplier', type=float, default=0.1, help='Noise multiplier for DP')

    # Model parameters
    parser.add_argument('--model', type=str, default='eegnet')
    parser.add_argument('--sample_rate', type=int, default=250)
    parser.add_argument('--F1', type=int, default=8)
    parser.add_argument('--D', type=int, default=2)
    parser.add_argument('--F2', type=int, default=16)
    parser.add_argument('--class_num', type=int, default=4)
    parser.add_argument('--channels', type=int, default=22)
    parser.add_argument('--samples', type=int, default=1001)    
    parser.add_argument('--dropout', type=float, default=0.5)

    # Basic training setup
    parser.add_argument('--data_path', type=str, default='./data/BNCI2014001')
    parser.add_argument('--sub_id', type=str, default='1,2,3,4,5,6,7,8,9')
    parser.add_argument('--output_path', type=str, default='./output')
    parser.add_argument('--ea', type=lambda x:bool(strtobool(x)), default=True, help='Euclidean Alignment')
    
    # Federated training setup
    parser.add_argument('--global_epochs', type=int, default=200)
    parser.add_argument('--sample_num', type=int, default=4, help='Clients sampled per round')
    parser.add_argument('--local_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--early', type=lambda x:bool(strtobool(x)), default=False, help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=50)

    # FedBS specific
    parser.add_argument('--fedbs', type=lambda x:bool(strtobool(x)), default=False, help='Enable FedBS')
    parser.add_argument('--rho', type=float, default=0.1, help='Rho for FedBS (SAM)')

    args = parser.parse_args()

    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = f"{args.output_path}/save_models/{timestamp}"
    os.makedirs(save_path, exist_ok=True)

    print('='*90)
        
    Server_TestAcc_List = []
    Attack_Success_List = []
    subject_id = [int(i) for i in args.sub_id.split(',')]

    # Leave-One-Subject-Out (LOSO) Loop
    for id in subject_id:
        server_subject_id = [id]
        client_subject_id = [x for x in subject_id if x != id]

        print('-'*64)
        print(f'Server Subject {server_subject_id} Begin')
        print(f'Client subject ID: {client_subject_id}')
        
        train(args, server_subject_id, client_subject_id, Server_TestAcc_List, trace_func=tqdm.write, save_path=f'{save_path}/Model_ServerSub{id}.pth', Attack_Success_List=Attack_Success_List)
        
        print(f'Server Subject {server_subject_id} Complete')
        print('-'*64)

    # Final Results Logging & Metric Computation
    accs = np.array(Server_TestAcc_List, dtype=float)
    mean_acc = round(float(np.mean(accs)), 2)
    median_acc = round(float(np.median(accs)), 2)

    error_ratios = np.round(100.0 - accs, 2)
    mean_err = round(float(np.mean(error_ratios)), 2)
    median_err = round(float(np.median(error_ratios)), 2)

    columns = [int(i) for i in args.sub_id.split(',')] + ['Mean', 'Median']

    acc_row = list(accs) + [mean_acc, median_acc]
    err_row = list(error_ratios) + [mean_err, median_err]
    err_ratio_dec_row = [round(e / 100.0, 4) for e in err_row]

    rows = [acc_row, err_row, err_ratio_dec_row]
    index = ['Test Acc (%)', 'Error Ratio (%)', 'Error Ratio (0-1)']

    if Attack_Success_List:
        att_arr = np.array(Attack_Success_List, dtype=float)
        mean_att = round(float(np.mean(att_arr)), 2)
        median_att = round(float(np.median(att_arr)), 2)
        att_row = list(att_arr) + [mean_att, median_att]
        rows.append(att_row)
        index.append('Attack Success (%)')

    df = pd.DataFrame(rows, columns=columns, index=index)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}" if abs(x) >= 1 else f"{x:.4f}")

    dp_status = f"ON  (noise={args.noise_multiplier}, clip={args.clip_bound})" if args.dp else "OFF"
    fedbs_status = f"ON  (rho={args.rho})" if args.fedbs else "OFF"

    print('\n' + '='*80)
    print('                           FINAL EVALUATION RESULTS')
    print('='*80)
    print(f'  Run config:  global_epochs={args.global_epochs}  local_epochs={args.local_epochs}')
    print(f'               FedBS={fedbs_status}')
    print(f'               Differential Privacy={dp_status}')
    print(f'  Subjects:    {subject_id}')
    print('-'*80)
    print('\n[Per-Subject Matrix]')
    print(df.to_string())
    print('\n' + '-'*80)
    print('Aggregate Metrics:')
    print(f'  Mean Accuracy:        {mean_acc:.2f}%')
    print(f'  Median Accuracy:      {median_acc:.2f}%')
    print(f'  Mean Error Ratio:     {mean_err:.2f}%  ({mean_err/100:.4f})')
    print(f'  Median Error Ratio:   {median_err:.2f}%  ({median_err/100:.4f})')
    if Attack_Success_List:
        print(f'  Mean Attack Success:  {mean_att:.2f}%')
        print(f'  Median Attack Success:{median_att:.2f}%')
    print('='*80 + '\n')

    # Save full matrix to CSV
    df.to_csv(f"{save_path}/evaluation_results.csv")