import numpy
import torch
import os.path as osp

from scipy.sparse import csr_matrix
from torch_geometric.utils import to_scipy_sparse_matrix, from_scipy_sparse_matrix, to_dense_adj, dense_to_sparse

import lib.losses as L
import lib.augmentation as A
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from torch.optim import Adam

from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

import argparse
import configparser
from datetime import datetime
import copy
import csv

from lib.eval import get_split, SVMEvaluator
from lib.logger import get_logger
from lib.losses import InfoNCE
from lib.utils import init_seed
from lib.utils import normalize_batch_embedding


from model.encoder_new import *
from model.gnn_models import *
from model.models.contrast_model import *


def val_epoch(aug_encoder_model, encoder_model, contrast_model, dataloader, epoch, args):
    encoder_model.eval()
    contrast_model.eval()
    total_val_loss = 0
    val_per_epoch = len(dataloader)
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(args.device)
            optimizer.zero_grad()
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            mini_loader = DataLoader(data, batch_size=1)
            new_edge_index = torch.Tensor([[0], [0]])
            edge_index_cum = 0
            for mini_idx, mini in enumerate(mini_loader):
                new_adj_matrix, _, _ = aug_encoder_model(mini.x, mini.edge_index, mini.batch)
                edge_mask = torch.round(new_adj_matrix).cpu().detach().to(args.device).bool()
                # print(torch.add(mini.edge_index[:, edge_mask[0, :]], edge_index_cum))
                new_edge_index = torch.cat(
                    (new_edge_index.to(args.device), (mini.edge_index[:, edge_mask[:, 0]] + edge_index_cum).int()), 1)
                edge_index_cum = edge_index_cum + mini.edge_index.size()[1] - 1

            # end
            # print(csr_matrix(numpy.array(adj_matrix.cpu())[0]))
            if new_edge_index.size()[1] == 0:
                print("Model Collapsed!")
                new_edge_index = torch.round(torch.Tensor([[0], [0]])).type(torch.int64).to(args.device)
                collapse = True
            if new_edge_index.max().item() <= data.x.size()[0]:
                z, g, z1, z2, g1, g2 = encoder_model(data.x, data.edge_index,
                                                    new_edge_index.cpu().type(torch.LongTensor).to(args.device),
                                                    data.batch)
                if args.contrast_mode == 'G2G':
                    g_normed = normalize_batch_embedding(g)
                    g_flag = torch.where(g_normed > args.threshold, 0, 1)
                    g2_pos = g_flag * g2
                    g1, g2, g2_pos = [encoder_model.encoder.project(g_) for g_ in [g1, g2, g2_pos]]
                    # g1.shape: [batch_size, num_embeddings]
                    loss = contrast_model(g1=g1, g2=g2, g2_pos=g2_pos, batch=data.batch)

            if args.contrast_mode == 'G2G':
                g1, g2 = [encoder_model.encoder.project(g_) for g_ in [g1, g2]]
                # g1.shape: [batch_size, num_embeddings]
                loss = contrast_model(g1=g, g2=g2, batch=data.batch)
            total_val_loss += loss.item()
            if batch_idx % args.log_step == 0: 
                logger.info('Val Epoch {}: {}/{} Loss: {:.6f}'.format(
                    epoch, batch_idx, val_per_epoch, loss.item()
                ) )
    val_loss = total_val_loss / val_per_epoch
    logger.info('******* Val Epoch {}: average loss: {:.6f}'.format(epoch, val_loss))
    return val_loss
def train_epoch(aug_encoder_model, encoder_model, contrast_model, dataloader, aug_optimizer, optimizer, epoch, args):
    aug_encoder_model.train()
    encoder_model.train()
    contrast_model.train()
    total_epoch_loss = 0
    train_per_epoch = len(dataloader)
    collapse = False
    for batch_idx, data in enumerate(dataloader):
        data = data.to(args.device)
        optimizer.zero_grad()
        aug_optimizer.zero_grad()
        #print(data.batch.size())

        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
        #begin
        mini_loader = DataLoader(data, batch_size=1)
        new_edge_index = torch.Tensor([[0],[0]])
        edge_index_cum = 0
        for mini_idx, mini in enumerate(mini_loader):
                #edge_mask = torch.round(new_adj_matrix).cpu().detach().to(args.device).bool()
            new_adj_matrix, g1, g2 = aug_encoder_model(mini.x, mini.edge_index,mini.batch)
            edge_mask = torch.round(new_adj_matrix).cpu().detach().to(args.device).bool()
            new_edge_index = torch.cat(
                (new_edge_index.to(args.device), (mini.edge_index[:, edge_mask[:, 0]] + edge_index_cum).int()), 1)
            edge_index_cum = edge_index_cum + mini.edge_index.size()[1] - 1
            l1 = -nn.MSELoss()(torch.ones_like(new_adj_matrix), new_adj_matrix)
            if (mini.x != None):
                g1, g2 = [encoder_model.encoder.project(g_) for g_ in [g1, g2]]
                l2 = nn.MSELoss()(normalize_batch_embedding(g1),normalize_batch_embedding(g2))
                loss = l1 + l2*1e10
            else:
                loss = l1
            loss.backward()
            aug_optimizer.step()
            aug_optimizer.zero_grad()
        #end
        #print(csr_matrix(numpy.array(adj_matrix.cpu())[0]))
        if new_edge_index.size()[1] == 0:
            print("Model Collapsed!")
            new_edge_index = torch.round(torch.Tensor([[0],[0]])).type(torch.int64).to(args.device)
            collapse = True
        #print(data.edge_index,new_edge_index)
        if new_edge_index.max().item() <= data.x.size()[0]:
            z, g, z1, z2, g1, g2 = encoder_model(data.x, data.edge_index, new_edge_index.cpu().type(torch.LongTensor).to(args.device), data.batch)
            if args.contrast_mode == 'G2G':
                g_normed = normalize_batch_embedding(g)
                g_flag = torch.where(g_normed > args.threshold, 0, 1)
                g2_pos = g_flag * g2
                g1, g2, g2_pos = [encoder_model.encoder.project(g_) for g_ in [g1, g2, g2_pos]]
                # g1.shape: [batch_size, num_embeddings]
                loss = contrast_model(g1=g1, g2=g2,g2_pos=g2_pos ,batch=data.batch)
            loss.backward()
            optimizer.step()

        total_epoch_loss += loss.item()
        if batch_idx % args.log_step == 0:
            print('Train Epoch {}: {}/{} Loss: {:.6f}'.format(
                epoch, batch_idx, train_per_epoch, loss.item()
            ))
    train_epoch_loss = total_epoch_loss / train_per_epoch
    print('*******Train Epoch {}: average Loss: {:.6f}'.format(epoch, train_epoch_loss))
    return train_epoch_loss, collapse


def test(aug_encoder_model, encoder_model, dataloader, args):
    aug_encoder_model.eval()
    encoder_model.eval()
    x = []
    y = []
    encoder_model.eval()
    with torch.no_grad():
        for data in dataloader:
            data = data.to(args.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            mini_loader = DataLoader(data, batch_size=1)
            new_edge_index = torch.Tensor([[0], [0]])
            edge_index_cum = 0
            for mini_idx, mini in enumerate(mini_loader):
                new_adj_matrix, _, _ = aug_encoder_model(mini.x, mini.edge_index, mini.batch)
                edge_mask = torch.round(new_adj_matrix).cpu().detach().to(args.device).bool()
                # print(torch.add(mini.edge_index[:, edge_mask[0, :]], edge_index_cum))
                new_edge_index = torch.cat(
                    (new_edge_index.to(args.device), (mini.edge_index[:, edge_mask[:, 0]] + edge_index_cum).int()), 1)
                edge_index_cum = edge_index_cum + mini.edge_index.size()[1] - 1

            # end
            # print(csr_matrix(numpy.array(adj_matrix.cpu())[0]))
            if new_edge_index.size()[1] == 0:
                print("Model Collapsed!")
                new_edge_index = torch.round(torch.Tensor([[0], [0]])).type(torch.int64).to(args.device)
                collapse = True
            # print(data.edge_index,new_edge_index)
            if new_edge_index.max().item() <= data.x.size()[0]:
                z, g, z1, z2, g1, g2 = encoder_model(data.x, data.edge_index,
                                                     new_edge_index.cpu().type(torch.LongTensor).to(args.device),
                                                     data.batch)
                #if args.contrast_mode == 'G2G':
                    #g_normed = normalize_batch_embedding(g)
                    #g_flag = torch.where(g_normed > args.threshold, 0, 1)
                    #g2_pos = g_flag * g2
                    #g1, g2, g2_pos = [encoder_model.encoder.project(g_) for g_ in [g1, g2, g2_pos]]
                    # g1.shape: [batch_size, num_embeddings]
                    #loss = contrast_model(g1=g1, g2=g2, g2_pos=g2_pos, batch=data.batch)

                x.append(g)
                y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)

    split = get_split(num_samples=x.size()[0], train_ratio=0.8, test_ratio=0.1)
    result = SVMEvaluator(linear=False)(x, y, split)
    return result



#*************************************************************#
MODE = 'train'
DEBUG = 'True'
DATASET = 'NCI1'
DEVICE = 'cuda:1'
print(DEVICE)

# get configuration
config_file = 'GCL.conf'
print('Read Configuration file: %s' % (config_file))
config = configparser.ConfigParser()
config.read(config_file)
#parser
args = argparse.ArgumentParser(description='arguments')
# args.add_argument('--dataset', default=DATASET,type=str)
args.add_argument('--mode', default=MODE, type=str)
args.add_argument('--device', default=config['log']['device'], type=str, help='indices of GPUs')
args.add_argument('--debug', default=config['log']['debug'], type=eval)
args.add_argument('--cuda', default=True, type=bool)


#data
args.add_argument('--dataset1', default=config['data']['dataset1'], type=str)
args.add_argument('--dataset2', default=config['data']['dataset2'], type=str)
args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)


#contrast_model
args.add_argument('--aug1', default=config['GCL']['aug1'], type=str)
args.add_argument('--aug2', default=config['GCL']['aug2'], type=str)
args.add_argument('--p', default=config['GCL']['p'], type=float)
args.add_argument('--gnn', default=config['GCL']['gnn'], type=str)
args.add_argument('--gnn_hidden_dim', default=config['GCL']['gnn_hidden_dim'], type=int)
args.add_argument('--gnn_output_dim', default=config['GCL']['gnn_output_dim'], type=int)
args.add_argument('--gnn_layer', default=config['GCL']['gnn_layer'], type=int)
args.add_argument('--contrast_mode', default=config['GCL']['contrast_mode'], type=str)
args.add_argument('--threshold', default=config['GCL']['threshold'], type=float)
args.add_argument('--is_comp', default=config['GCL']['is_comp'], type=eval)
#Tain
args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
args.add_argument('--tau', default=config['train']['tau'], type=float)
args.add_argument('--seed', default=config['train']['seed'], type=int)
args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
args.add_argument('--epochs', default=config['train']['epochs'], type=int)
args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
args.add_argument('--remark', default=None, type=str)

#log
args.add_argument('--log_dir', default='experiments_new_1', type=str)
args.add_argument('--log_step', default=config['log']['log_step'], type=int)
args.add_argument('--plot', default=config['log']['plot'], type=eval)

args = args.parse_args()

init_seed(args.seed)


print(args.device)

# path = osp.join(osp.expanduser('~'), 'datasets')
print(args.dataset1)
dataset = TUDataset('./data', name=args.dataset1)
num_data = len(dataset)
train_set = dataset
# train_set = dataset[: int( num_data * (1-args.val_ratio-args.test_ratio))]
val_set = dataset[int(num_data * (1-args.val_ratio-args.test_ratio))+1 : ]

train_loader = DataLoader(train_set, batch_size=args.batch_size)
# val_loader = DataLoader(val_set, batch_size=args.batch_size)
val_loader = train_loader
#print(len(train_loader))
# dataloader = DataLoader(dataset, batch_size=args.batch_size)
# val_loader = DataLoader(dataset, batch_size=args.batch_size)
# val_loader = None
test_loader = None
input_dim = max(dataset.num_features, 1)
input_dim_aug = max(dataset.num_edge_labels, 1)
mini_loader = DataLoader(dataset, batch_size=1)
for mini_idx, mini in enumerate(mini_loader):
    if mini.edge_index.size()[1] > input_dim_aug:
        input_dim_aug = mini.edge_index.size()[1]
        print(input_dim_aug)
#for batch_idx, data in enumerate(train_loader):
#    mini_loader = DataLoader(data, 1)
#    for mini_dix, mini in enumerate(mini_loader):
#        if mini.edge_index.size()[1] > input_dim_aug:
#            input_dim_aug = data.edge_index.size()[1]
#print("input_dim",input_dim_aug)
# augmentation
aug1 = A.generate_aug(args.aug1, p=args.p)
aug2 = A.generate_aug(args.aug2, p=args.p)
# aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                        # A.NodeDropping(pn=0.1),
                        # A.FeatureMasking(pf=0.1),
                        # A.EdgeRemoving(pe=0.1)], 1)




if args.gnn == 'GIN':
    gnn = GConv(input_dim, args.gnn_hidden_dim, args.gnn_layer).to(args.device)
    aug_gnn = GConv(input_dim, args.gnn_hidden_dim, args.gnn_layer).to(args.device)

# gconv = GConv(input_dim=input_dim, hidden_dim=32, num_layers=2).to(device)
aug_encoder_model = Aug_Encoder(input_dim = input_dim_aug, encoder=aug_gnn, augmentor=(aug1, aug2), args=args).to(args.device)
encoder_model = Encoder(encoder=gnn, augmentor=(aug1, aug2)).to(args.device)

if args.loss_func == 'InfoNCE':
    loss = L.InfoNCE(args.tau)
elif args.loss_func == 'BiInfoNCE':
    loss = L.BiInfoNCE(args.tau, args.threshold)
elif args.loss_func == 'DGCL':
    loss = L.DGCLloss(args.tau)

contrast_model = DualBranchContrast(loss=loss, mode=args.contrast_mode).to(args.device)

optimizer = Adam(encoder_model.parameters(), lr=args.lr_init)
aug_optimizer = Adam(aug_encoder_model.parameters(), lr=args.lr_init/10, weight_decay=args.lr_init/100)
aug_optimizer.add_param_group({"params": aug_encoder_model.mask})
#aug_optimizer.add_param_group({"params": aug_encoder_model.mlp_edge_model})
#print(optimizer.param_groups[0])
#optimizer.add_param_group({"params": aug_encoder_model.threshold})
#optimizer.add_param_group({"params": aug_encoder_model.prob})

#Config log path
current_time = datetime.now().strftime('%m月%d日%H时%M分%S秒')
current_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(current_dir, args.log_dir, args.dataset1, current_time)
logger = get_logger(log_dir, debug=args.debug)
logger.info(args.remark)
for arg, value in sorted(vars(args).items()):
    logger.info("%s: %r", arg, value)


#Begin to train
train_loss_list = []
val_loss_list = []
best_loss = float('inf')


best_contrast_model = None
best_encoder_model = None
not_improved_count = 0


for epoch in range(args.epochs):
    loss, collapse = train_epoch(aug_encoder_model, encoder_model, contrast_model, train_loader, aug_optimizer, optimizer, epoch, args)
    if val_loader == None:
        val_loader = train_loader
    val_epoch_loss = val_epoch(aug_encoder_model, encoder_model, contrast_model, val_loader, epoch, args)

    train_loss_list.append(loss)
    val_loss_list.append(val_epoch_loss)
    if collapse == True:
        not_improved_count = 1000000
    if loss > 1e6:
        logger.info('Gradient explosion detected. Ending...')
        break
    if val_epoch_loss < best_loss:
        best_loss = val_epoch_loss
        not_improved_count = 0
        best_state = True
    else: 
        not_improved_count += 1
        best_state = False
    
    # early stop
    if args.early_stop: 
        if not_improved_count >= args.early_stop_patience:
            logger.info("Validation Performance didn\'t improve for {} epochs. "
                        "Training Stops.".format(args.early_stop_patience)
            )
            break
    if best_state == True:
        logger.info('************Current best model saved!')
        best_contrast_model = copy.deepcopy(contrast_model.state_dict())
        best_encoder_model = copy.deepcopy(encoder_model.state_dict())
        best_aug_model = copy.deepcopy(aug_encoder_model.state_dict())
    
    # save the best model to file
    if not args.debug:
        torch.save(best_contrast_model, os.path.join(log_dir,'best_contrast_model.pth'))
        torch.save(best_encoder_model, os.path.join(log_dir, 'best_encoder_model.pth'))
        torch.save(best_aug_model, os.path.join(log_dir, 'best_aug_model.pth'))
if args.early_stop:
    contrast_model.load_state_dict(best_contrast_model)
    encoder_model.load_state_dict(best_encoder_model)
    aug_encoder_model.load_state_dict(best_aug_model)
test_result = test(aug_encoder_model, encoder_model, train_loader, args)
logger.info(f'(E): Best test F1Mi={test_result["micro_f1"]:.4f}, F1Ma={test_result["macro_f1"]:.4f}')


exp_summary_file = os.path.join(current_dir, args.log_dir, args.dataset1,'summary.csv')
with open(exp_summary_file, 'a') as f:
    writter = csv.writer(f)
    writter.writerow([args.aug1, args.aug2,args.p ,test_result["micro_f1"]])