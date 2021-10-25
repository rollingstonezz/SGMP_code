import argparse
import numpy as np
import os
from tqdm import tqdm
import time
import pickle as pkl
import json

import torch
import copy
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter

from utils.utils import build_spanning_tree_edge, find_higher_order_neighbors, add_self_loops
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--model', type=str, default='SGMP')
    parser.add_argument('--dataset', type=str, default='BACE')
    parser.add_argument('--split', type=str, default='811')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--readout', type=str, default='add')
    parser.add_argument('--spanning_tree', type=str, default='False')
    parser.add_argument('--structure', type=str, default='sc')

    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--random_seed_2', type=int, default=12345)
    parser.add_argument('--label', type=int, default=12)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--test_per_round', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--cutoff', type=float, default=10.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    args = parser.parse_args()

    return args

def load_data(args):
    if args.dataset == 'synthetic':
        with open(os.path.join(args.data_dir, 'synthetic.pkl'), 'rb') as file:
            dataset = pkl.load(file)
        dataset = dataset[torch.randperm(len(dataset))]    
        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )

    elif args.dataset == 'QM9':
        from torch_geometric.datasets import QM9
        dataset = QM9(root=os.path.join(args.data_dir, 'QM9'))
        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(random_state.permutation(np.arange(130831)))
        dataset = dataset[perm]
        train_valid_split, valid_test_split = 110000, 10000

    elif args.dataset == 'brain':
        from utils.brain_load_data import load_brain_data
        dataset = load_brain_data(data_dir=args.data_dir, structure='sc', threshold=5e5, random_seed=args.random_seed) 
        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )

    elif args.dataset in ["ESOL", "Lipo", "BACE", "BBBP"]:
        from utils.moleculenet import MoleculeNet
        dataset = MoleculeNet(root=os.path.join(args.data_dir, 'MoleculeNet', args.dataset),name=args.dataset)
        dataset = dataset[torch.randperm(len(dataset))]    
        train_valid_split = int( int(args.split[0]) / 10 * len(dataset) )
        valid_test_split = int( int(args.split[1]) / 10 * len(dataset) )
    else:
        raise Exception('Dataset not recognized.')

    
    train_dataset = dataset[:train_valid_split]
    valid_dataset = dataset[train_valid_split:train_valid_split+valid_test_split]
    test_dataset = dataset[train_valid_split+valid_test_split:]
        
    print('======================')
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of valid graphs: {len(valid_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def main(data, args):
    # logging
    if args.spanning_tree == 'True':
        LOG_DIR = os.path.join(args.save_dir, args.dataset, args.model+'_st')
    else:
        LOG_DIR = os.path.join(args.save_dir, args.dataset, args.model)
    if args.dataset == 'QM9':
        LOG_DIR = os.path.join(LOG_DIR, str(args.label))
        
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    log_file = os.path.join(LOG_DIR, 'log.csv')
    result_file = os.path.join(LOG_DIR, 'results.txt')
    
    # device
    if args.device == 'cpu':
        device = torch.device("cpu")
    elif args.device == 'gpu':
        device = torch.device("cuda:0")
    else:
        raise Exception('Please assign the type of device: cpu or gpu.')
    
    if args.dataset == 'brain':
        input_channels_node, hidden_channels, readout = 1, 64, args.readout
    elif args.dataset == 'QM9':
        input_channels_node, hidden_channels, readout = 11, 64, args.readout
    else:
        input_channels_node, hidden_channels, readout = 9, 64, args.readout
    
    if args.dataset in [ 'BACE', 'BBBP']:
        task = 'classification'
    elif args.dataset in ['QM9', 'ESOL', 'Lipo']:
        task = 'regression'
    elif args.dataset in ['brain']:
        task = 'regression'
            
    if task == 'regression':
        output_channels = 1
    else:
        output_channels = 2
        
    # select model and its parameter
    print(args.model)
    if args.model == 'GIN':
        from models.GIN import GINNet
        net = GINNet(input_channels_node, hidden_channels, output_channels, readout=readout, eps=0., num_layers=args.num_layers)
    elif args.model == 'GAT':
        from models.GAT import GATNet
        net = GATNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'GatedGraphConv':
        from models.GatedGraphConv import GatedNet
        net = GatedNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'PointNet':
        from models.PointNet import PointNet
        net = PointNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'PPFNet':
        from models.PPFNet import PPFNet
        net = PPFNet(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'SGCN':
        from models.SGCN import SGCN
        net = SGCN(input_channels_node, hidden_channels, output_channels, readout=readout, num_layers=args.num_layers)
    elif args.model == 'Schnet':
        from models.Schnet import Schnet
        net = Schnet(input_channels_node=input_channels_node, 
                    hidden_channels=hidden_channels, output_channels=output_channels, num_interactions=args.num_layers,
                    num_gaussians=hidden_channels, cutoff=args.cutoff, readout=readout)
    elif args.model == 'Dimenet':
        from models.Dimenet import Dimenet
        net = Dimenet(input_channels_node=input_channels_node, 
                    hidden_channels=hidden_channels, output_channels=output_channels, num_blocks=args.num_layers,
                    cutoff=args.cutoff)
    elif args.model == 'SGMP':
        from models.SGMP import SGMP
        net = SGMP(input_channels_node=input_channels_node, 
            hidden_channels=hidden_channels, output_channels=output_channels,
            num_interactions=args.num_layers, cutoff=args.cutoff,
            readout=readout)
        
    model = net.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=5, min_lr=5e-5)
    
    if task == 'regression':
        criterion = torch.nn.MSELoss()
        from sklearn.metrics import mean_absolute_error
        measure = mean_absolute_error
    elif task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
        from sklearn.metrics import accuracy_score
        measure = accuracy_score
        
    # get train/valid/test data
    train_loader, valid_loader, test_loader = data        
        
    def train(loader, model, args):
        model.train()
        for data in (loader):  # Iterate in batches over the training dataset.
            x, pos, edge_index, batch = data.x.float(), data.pos, data.edge_index, data.batch
            if args.dataset == 'brain':
                x = x.long()
            if args.dataset == 'QM9':
                y = data.y[:, args.label]
            else:
                y = data.y.long() if task == 'classification' else data.y
            num_nodes = data.num_nodes
            num_edges = data.num_edges
            if args.spanning_tree == 'True':
                edge_index = build_spanning_tree_edge(edge_index.cpu(), algo='scipy', num_nodes=num_nodes, num_edges=num_edges)
            x, pos, edge_index, batch, y = x.to(device), pos.to(device), edge_index.to(device), batch.to(device), y.to(device)
            if args.model == 'SGMP':
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes) # add self loop to avoid crash on specific data point with longest path < 3
                _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(edge_index, num_nodes, order=3)
                out = model(x, pos, batch, edge_index_3rd)
            else:
                out = model(x, pos, edge_index, batch)
                
            if task == 'classification':
                loss = criterion(out, y.reshape(-1))  # Compute the loss.
            else:
                loss = criterion(out.reshape(-1, 1), y.reshape(-1, 1))  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader, model, args):
        model.eval()
        y_hat, y_true = [], []
        loss_total, total_graph = 0, 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            x, pos, edge_index, batch = data.x.float(), data.pos, data.edge_index, data.batch
            if args.dataset == 'brain':
                x = x.long()
            if args.dataset == 'QM9':
                y = data.y[:, args.label]
            else:
                y = data.y.long() if task == 'classification' else data.y
            num_nodes = data.num_nodes
            num_edges = data.num_edges
            if args.spanning_tree == 'True':
                out_list = []
                num_samples = 25
                for _ in range(num_samples): # mote-carlo to reduce the variance on test
                    sp_edge_index = build_spanning_tree_edge(edge_index.cpu(), algo='scipy', num_nodes=num_nodes, num_edges=num_edges)
                    x, pos, sp_edge_index, batch, y = x.to(device), pos.to(device), sp_edge_index.to(device), batch.to(device), y.to(device)
                    if args.model == 'SGMP':
                        sp_edge_index, _ = add_self_loops(sp_edge_index, num_nodes=num_nodes, fill_value=-1.)
                        _, _, edge_index_3rd, _, _, _, _, _ = find_higher_order_neighbors(sp_edge_index, num_nodes, order=3)
                        out = model(x, pos, batch, edge_index_3rd)
                    else:
                        out = model(x, pos, edge_index, batch)
                    out_list.append(out.detach().cpu())

                if task == 'classification':
                    out = torch.cat(out_list,dim=0).reshape(num_samples,-1,2).mean(dim=0).to(device)
                    loss = criterion(out, y.reshape(-1))  # Compute the loss.
                else:
                    out = torch.cat(out_list,dim=0).reshape(num_samples,-1,1).mean(dim=0).to(device)
                    loss = criterion(out.reshape(-1, 1), y.reshape(-1, 1))  # Compute the loss.
                loss_total += loss.detach().cpu() * data.num_graphs
                total_graph += data.num_graphs
                if task == 'classification':
                    pred = out.argmax(dim=1)  # Use the class with highest probability.
                    y_hat += list(pred.cpu().detach().numpy().reshape(-1))
                else:
                    y_hat += list(out.cpu().detach().numpy().reshape(-1))
                y_true += list(y.cpu().detach().numpy().reshape(-1))

        return loss_total/total_graph, y_hat, y_true
            
    with open(log_file, 'a') as f:
        print(f"Epoch, Valid loss, Valid score, --- %s seconds ---", file=f) 

    start_time = time.time()
    best_valid_score = 1e10 if task == 'regression' else 0
    best_model = None
    for epoch in (range(1, args.epoch)):
        # training
        train(train_loader, model, args)
        
        if epoch % args.test_per_round == 0:
            valid_loss, yhat_valid, ytrue_valid = test(valid_loader, model, args)
            valid_score = measure(ytrue_valid, yhat_valid)

            if epoch >= 100:
                lr = scheduler.optimizer.param_groups[0]['lr']
                scheduler.step(valid_loss)

            with open(log_file, 'a') as f:
                print(f"{epoch:03d}, {valid_loss:.4f}, {valid_score:.4f} ,{(time.time() - start_time):.4f}", file=f) 

            if task == 'regression':
                if valid_score < best_valid_score:
                    best_valid_score = valid_score
                    best_model = copy.deepcopy(model)
            else:
                if valid_score > best_valid_score:
                    best_valid_score = valid_score
                    best_model = copy.deepcopy(model)
                    
    train_loss, yhat_train, ytrue_train = test(train_loader, model, args)
    train_score = measure(ytrue_train, yhat_train)
    valid_loss, yhat_valid, ytrue_valid = test(valid_loader, model, args)
    valid_score = measure(ytrue_valid, yhat_valid) 
    test_loss, yhat_test, ytrue_test = test(test_loader, model, args)
    test_score = measure(ytrue_test, yhat_test)
    with open(result_file, 'a') as f:
        if task == 'regression':
            print(f"Final, Train RMSE: {np.sqrt(train_loss):.4f}, Train MAE: {train_score:.4f}, Valid RMSE: {np.sqrt(valid_loss):.4f}, Valid MAE: {valid_score:.4f}, Test RMSE: {np.sqrt(test_loss):.4f}, Test MAE: {test_score:.4f}", file=f) 
        elif task == 'classification':
            print(f"Final, Train loss: {train_loss:.4f}, Train acc: {train_score:.4f}, Valid loss: {valid_loss:.4f}, Valid acc: {valid_score:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_score:.4f}", file=f) 
        
    
    train_loss, yhat_train, ytrue_train = test(train_loader, best_model, args)
    train_score = measure(ytrue_train, yhat_train)
    valid_loss, yhat_valid, ytrue_valid = test(valid_loader, best_model, args)
    valid_score = measure(ytrue_valid, yhat_valid) 
    test_loss, yhat_test, ytrue_test = test(test_loader, best_model, args)
    test_score = measure(ytrue_test, yhat_test) 
    with open(result_file, 'a') as f:
        if task == 'regression':
            print(f"Best Model, Train RMSE: {np.sqrt(train_loss):.4f}, Train MAE: {train_score:.4f}, Valid RMSE: {np.sqrt(valid_loss):.4f}, Valid MAE: {valid_score:.4f}, Test RMSE: {np.sqrt(test_loss):.4f}, Test MAE: {test_score:.4f}") 
            print(f"Best Model, Train RMSE: {np.sqrt(train_loss):.4f}, Train MAE: {train_score:.4f}, Valid RMSE: {np.sqrt(valid_loss):.4f}, Valid MAE: {valid_score:.4f}, Test RMSE: {np.sqrt(test_loss):.4f}, Test MAE: {test_score:.4f}", file=f) 
        elif task == 'classification':
            print(f"Best Model, Train loss: {train_loss:.4f}, Train acc: {train_score:.4f}, Valid loss: {valid_loss:.4f}, Valid acc: {valid_score:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_score:.4f}") 
            print(f"Best Model, Train loss: {train_loss:.4f}, Train acc: {train_score:.4f}, Valid loss: {valid_loss:.4f}, Valid acc: {valid_score:.4f}, Test loss: {test_loss:.4f}, Test acc: {test_score:.4f}", file=f) 
            
    
if __name__ == '__main__':
    args = get_args()
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    torch.manual_seed(args.random_seed)  
    data = load_data(args)
    main(data, args)
    
    
