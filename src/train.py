from __future__ import division
from __future__ import print_function

import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, train_test_split
from utils import load_data, get_links, dse_normalize, validation, binary_cross_entropy_loss, save_result, save_all
from model import HetAgg
import warnings
warnings.filterwarnings("ignore")

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='/content/drive/My Drive/Colab Notebooks/idse/data/', help='path to data')
parser.add_argument('--result_path', type=str, default='/content/drive/My Drive/Colab Notebooks/idse/result/', help='path to result')
parser.add_argument('--model_path', type=str, default='/content/drive/My Drive/Colab Notebooks/idse/model/', help='path to save model')
parser.add_argument('--D_n', type=int, default=1020, help='number of drug node')
parser.add_argument('--S_n', type=int, default=5599, help='number of side-effect node')
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-3, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=1024, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=10, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.02, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=500, help='Patience')

args = parser.parse_args()
print("------arguments-------")
for k, v in vars(args).items():
    print(k + ': ' + str(v))
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
fpt_feature, mpnn_feature = load_data()
data_set = get_links(path=args.data_path, dataset="drug_se_matrix.txt", D_n=args.D_n, S_n=args.S_n)


if args.cuda:
    fpt_feature = fpt_feature.cuda()
    mpnn_feature = mpnn_feature.cuda()


def train(model, optimizer, mask, target, train_idx, train_set):
    model.train()
    optimizer.zero_grad()
    outputs = model()
    output = torch.flatten(torch.mul(mask, outputs))
    loss_train = binary_cross_entropy_loss(output, target)
    loss_train.backward()
    optimizer.step()
    output = output[train_idx]
    noutput = torch.sigmoid(output).cpu().detach().numpy()
    metrics = validation(noutput, train_set)
    return loss_train.data.item(), metrics[0], metrics[1], outputs


def compute_test(test_set, outputs, mask, test_idx, flag=False):
    output = torch.flatten(torch.mul(mask, outputs))[test_idx]
    noutput = torch.sigmoid(output).cpu().detach().numpy()
    metrics = validation(noutput, test_set, flag)
    return metrics


kf = StratifiedKFold(n_splits=10, shuffle=True)
counter = 1
auc_arr = []
aupr_arr = []
mcc_arr = []
f1_arr = []
prec_arr = []
recall_arr = []
ap_arr = []
mr_arr = []
valid_aupr_arr = []


for train_index, test_index in kf.split(data_set, data_set):
    train_index, valid_index = train_test_split(train_index, test_size=0.05)
    train_set = data_set[train_index]
    valid_set = data_set[valid_index]
    print("train shape:", train_set.shape, ", valid shape:", valid_set.shape)
    test_set = data_set[test_index]
    print('Begin {}th folder'.format(counter),
          'train_size {}'.format(len(train_index)),
          'train_label {}'.format(np.sum(train_set)),
          'valid_label {}'.format(np.sum(valid_set)),
          'test_label {}'.format(np.sum(test_set)))
    
    train_mask = np.zeros(args.D_n * args.S_n)
    train_mask[train_index] = 1
    target = np.multiply(data_set, train_mask)
    matrix = target.reshape(args.D_n, args.S_n)
    
    print('train_mask {}'.format(np.sum(train_mask)),
            'matrix {}'.format(np.sum(matrix)))
    
    train_mask = torch.from_numpy(train_mask.reshape(args.D_n, args.S_n)).cuda()
    target = torch.from_numpy(target).cuda()

    drug_se_train, se_drug_train = dse_normalize(args.cuda, matrix, D_n=args.D_n, S_n=args.S_n)
    
    test_mask = np.zeros(args.D_n * args.S_n)
    test_mask[test_index] = 1
    test_mask = torch.from_numpy(test_mask.reshape(args.D_n, args.S_n)).cuda()

    valid_mask = np.zeros(args.D_n * args.S_n)
    valid_mask[valid_index] = 1
    valid_mask = torch.from_numpy(valid_mask.reshape(args.D_n, args.S_n)).cuda()
    
    model = HetAgg(args, args.dropout, mpnn_feature, fpt_feature, drug_se_train, se_drug_train)
    model.init_weights()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model.cuda()
    # Train model
    t_total = time.time()
    bad_counter = 0
    best_epoch = 0
    best_pr = 0
    final_outputs = []
    
    skf = StratifiedKFold(n_splits=100, shuffle=False) # batch
    
    for epoch in range(args.epochs):
        auc, aupr, outputs = [], [], []
        loss = 0
        t = time.time()
        loss, train_auc, train_aupr, outputs = train(model, optimizer, train_mask, target, train_index, train_set)
        print('folder: {}'.format(counter),
          'Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss),
          'train_auc: {:.4f}'.format(train_auc),
          'train_aupr: {:.4f}'.format(train_aupr),
          'time: {:.4f}s'.format(time.time() - t),
          'lr:', optimizer.defaults['lr'])
        valid_metrics = compute_test(valid_set, outputs, valid_mask, valid_index)
        valid_auc, valid_aupr = valid_metrics[0], valid_metrics[1]
        test_metrics = compute_test(test_set, outputs, test_mask, test_index)
        test_auc, test_aupr = test_metrics[0], test_metrics[1]
        print("Valid set results:",
              "folder= {}".format(counter),
              'Epoch: {:04d}'.format(epoch+1),
              'valid_auc: {:.4f}'.format(valid_auc),
              'valid_aupr: {:.4f}'.format(valid_aupr))
        print("Test set results:",
              "folder= {}".format(counter),
              'Epoch: {:04d}'.format(epoch+1),
              'test_auc: {:.4f}'.format(test_auc),
              'test_aupr: {:.4f}'.format(test_aupr),
              'Best_epoch: {:04d}'.format(best_epoch+1))
        if valid_aupr > best_pr:
            best_pr = valid_aupr
            best_epoch = epoch
            bad_counter = 0
            final_outputs = outputs
        else:
            bad_counter += 1
    
        if bad_counter >= args.patience:
            break
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('Loading {}th epoch'.format(best_epoch))

    # Testing
    # save_result(final_outputs, data_set, test_mask, counter)
    save_all(final_outputs, test_mask, counter)
    test_auc, test_aupr, prec, recall, mcc, f1, ap, mr = compute_test(test_set, final_outputs, test_mask, test_index, True)
    print("Test set results:",
          "folder= {}".format(counter),
          'test_auc: {:.4f}'.format(test_auc),
          'test_aupr: {:.4f}'.format(test_aupr),
          'test_prec: {:.4f}'.format(prec),
          'test_recall: {:.4f}'.format(recall),
          'test_mcc: {:.4f}'.format(mcc),
          'test_f1: {:.4f}'.format(f1),
          'test_ap: {:.4f}'.format(ap),
          'test_mr: {:.4f}'.format(mr))
    valid_aupr_arr.append(best_pr)
    auc_arr.append(test_auc)
    aupr_arr.append(test_aupr)
    mcc_arr.append(mcc)
    f1_arr.append(f1)
    prec_arr.append(prec)
    recall_arr.append(recall)
    ap_arr.append(ap)
    mr_arr.append(mr)
    np.savetxt(args.result_path + 'valid_aupr_avg', [counter, np.mean(np.array(valid_aupr_arr))])
    np.savetxt(args.result_path + 'auc_avg', [counter, np.mean(np.array(auc_arr))])
    np.savetxt(args.result_path + 'aupr_avg', [counter, np.mean(np.array(aupr_arr))])
    np.savetxt(args.result_path + 'mcc_avg', [counter, np.mean(np.array(mcc_arr))])
    np.savetxt(args.result_path + 'f1_avg', [counter, np.mean(np.array(f1_arr))])
    np.savetxt(args.result_path + 'prec_avg', [counter, np.mean(np.array(prec_arr))])
    np.savetxt(args.result_path + 'recall_avg', [counter, np.mean(np.array(recall_arr))])
    np.savetxt(args.result_path + 'ap_avg', [counter, np.mean(np.array(ap_arr))])
    np.savetxt(args.result_path + 'mr_avg', [counter, np.mean(np.array(mr_arr))])
    np.savetxt(args.result_path + 'valid_aupr', np.array(valid_aupr_arr))
    np.savetxt(args.result_path + 'auc', np.array(auc_arr))
    np.savetxt(args.result_path + 'aupr', np.array(aupr_arr))
    np.savetxt(args.result_path + 'mcc', np.array(mcc_arr))
    np.savetxt(args.result_path + 'f1', np.array(f1_arr))
    np.savetxt(args.result_path + 'prec', np.array(prec_arr))
    np.savetxt(args.result_path + 'recall', np.array(recall_arr))
    np.savetxt(args.result_path + 'ap', np.array(ap_arr))
    np.savetxt(args.result_path + 'mr', np.array(mr_arr))
    counter += 1
