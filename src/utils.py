import numpy as np
import random
import math
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.utils.data as data
import sys
import binascii
import pywt
from sklearn.metrics import roc_auc_score, matthews_corrcoef, recall_score, precision_score, precision_recall_curve, roc_curve, auc, f1_score, average_precision_score
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD, NMF
from scipy import sparse

    
def row_normalize(a_matrix):
    a_matrix = a_matrix.astype(float)
    row_sums = a_matrix.sum(axis=1) + 1e-12
    new_matrix = a_matrix / row_sums[:, np.newaxis]
    new_matrix[np.isnan(new_matrix) | np.isinf(new_matrix)] = 0.0
    return new_matrix
    
    
def standardization(data):
    mu = np.mean(data, axis=1, keepdims=True)
    sigma = np.std(data, axis=1, keepdims=True)
    return (data - mu) / sigma
    
    
def dse_normalize(cuda, drug_se, D_n=1020, S_n=5599):
    se_drug = drug_se.T
    drug_se_normalize = torch.from_numpy(row_normalize(drug_se)).float()
    se_drug_normalize = torch.from_numpy(row_normalize(se_drug)).float()
    if cuda:
        drug_se_normalize = drug_se_normalize.cuda()
        se_drug_normalize = se_drug_normalize.cuda()
    return drug_se_normalize, se_drug_normalize
    
    
def gen_adj(A):
    D = torch.pow(A.sum(1), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
    
    
def wavelet_encoder(seq):
    meta_drug = np.array(list(map(lambda x: int(x, 16), seq)))
    ca, cd = pywt.dwt(meta_drug, 'db1')
    drug_feature = ca / np.sum(ca)
    return drug_feature


def load_data(path="/content/drive/My Drive/Colab Notebooks/idse/data/", mpnn = "drugs_structure_MPNN_ToxCast.npy", fpt="drugs.fpt", num=1020):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(fpt))
    fpt_feature = np.zeros((num, 128))
    drug_file = open(path + fpt, "r")
    index = 0
    for line in drug_file:
        line = line.strip()
        if line == "":
            hex_arr = np.zeros(128)
        else:
            # hex_arr = list(map(lambda x: int(x, 16), line))
            hex_arr = wavelet_encoder(line)
        fpt_feature[index] = hex_arr
        index += 1
    fpt_feature = torch.FloatTensor(fpt_feature)#.normal_(0, 0.1)
    mpnn_feature = torch.FloatTensor(row_normalize(np.load(path + mpnn)))#.normal_(0, 0.1)
    return fpt_feature, mpnn_feature


def get_links(path="/content/drive/My Drive/Colab Notebooks/idse/data/", dataset="drug_se_matrix.txt", D_n=1020, S_n=5599):
    drug_se = np.loadtxt(path + dataset)
    data_set = drug_se.flatten()
    return data_set
    
    
def decomposite_feature(matrix, dim):
    model = TruncatedSVD(n_components=dim)
    WW = model.fit_transform(matrix)
    HH = model.components_
    WW = torch.from_numpy(WW).float().cuda()
    HH = torch.from_numpy(HH).float().cuda()
    print('ww:', WW.shape, ', hh:', HH.shape)
    return WW, HH.t()
    
    
def sample_links(data, seed, pos_count, neg_count):
    random.seed(seed)
    pos_list = []
    neg_list = []
    for data_tmp in data:
        if data_tmp[-1] == 1:
            pos_list.append(data_tmp)
        else:
            neg_list.append(data_tmp)
    pos_data = random.sample(pos_list, pos_count)
    neg_data = random.sample(neg_list, neg_count)
    return np.array(pos_data + neg_data)


def save_result(outputs, data_set, test_mask, fold, path="/content/drive/My Drive/Colab Notebooks/idse/result/", D_n=1020, S_n=5599):
    mask = torch.from_numpy(np.where(data_set.reshape(D_n, S_n) == 1, 0, 1)).cuda()
    matrix = torch.mul(torch.mul(outputs, mask), test_mask)
    result = []
    for i in range(D_n):
            for j in range(S_n):
                if matrix[i][j] != 0:
                    # print(matrix[i][j])
                    result.append([torch.sigmoid(matrix[i][j]).cpu().detach(), i, j])
    result.sort(key=lambda item: item[0], reverse=True)
    np.save(path + 'case_fold' + str(fold), result)
    

def save_all(final_outputs, test_mask, fold, path="/content/drive/My Drive/Colab Notebooks/idse/result/"):
    np.save(path + 'result' + str(fold), final_outputs.cpu().detach().numpy())
    np.save(path + 'mask' + str(fold), test_mask.cpu().detach().numpy())
    
    
def save_covid(outputs, path="/content/drive/My Drive/Colab Notebooks/idse/result/", D_n=5, S_n=5599):
    result = []
    for i in range(D_n):
            for j in range(S_n):
                if outputs[i][j] != 0:
                    # print(matrix[i][j])
                    result.append([torch.sigmoid(outputs[i][j]).cpu().detach(), i, j])
    result.sort(key=lambda item: item[0], reverse=True)
    np.save(path + 'case_covid19', result)


def binary_cross_entropy_loss(inputs, targets):
    criteria = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([40]).cuda())
    loss = criteria(inputs, targets)
    return loss


def validation(y_pre, y, flag=False):
    prec, recall, _ = precision_recall_curve(y, y_pre)
    pr_auc = auc(recall, prec)
    fpr, tpr, threshold = roc_curve(y, y_pre)
    roc_auc = auc(fpr, tpr)
    if flag:
        ap = average_precision_score(y, y_pre)
        mr = mrank(y, y_pre)
        y_predict_class = y_pre
        y_predict_class[y_predict_class > 0.5] = 1
        y_predict_class[y_predict_class <= 0.5] = 0
        prec = precision_score(y, y_predict_class)
        recall = recall_score(y, y_predict_class)
        mcc = matthews_corrcoef(y, y_predict_class)
        f1 = f1_score(y, y_predict_class)
        return roc_auc, pr_auc, prec, recall, mcc, f1, ap, mr
    return roc_auc, pr_auc, _, _, _, _, _, _
    
    
def mrank(y, y_pre):
    index = np.argsort(-y_pre)
    r_label = y[index]
    r_index = np.array(np.where(r_label == 1)) + 1
    reci_sum = np.sum(1 / r_index)
    reci_rank = np.mean(1 / r_index)
    return reci_sum


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
