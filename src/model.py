#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from itertools import *
from sklearn import preprocessing


class HetAgg(nn.Module):
    def __init__(self, args, dropout, mpnn_feature, fpt_feature, drug_se_train, se_drug_train):
        super(HetAgg, self).__init__()
        self.embed_d = args.hidden
        self.D_n = args.D_n
        self.S_n = args.S_n
        self.args = args
        self.dropout = dropout
        self.dim = int(args.hidden / 2)
        self.drug_dim = 617
        self.fpt_dim = 128
        
        self.Wd = nn.Linear(self.embed_d, self.dim)
        self.Ws = nn.Linear(self.embed_d, self.dim)
        self.bnd = nn.BatchNorm1d(self.embed_d)
        self.bns = nn.BatchNorm1d(self.embed_d)
        
        self.Gd = nn.Linear((self.embed_d + self.drug_dim + self.fpt_dim), self.embed_d)
        self.Gs = nn.Linear(2 * self.embed_d, self.embed_d)
        self.bd = nn.BatchNorm1d(self.embed_d + self.drug_dim + self.fpt_dim)
        self.bs = nn.BatchNorm1d(2 * self.embed_d)
        
        self.Gd2 = nn.Linear((self.embed_d + self.drug_dim + self.fpt_dim), self.embed_d)
        self.Gs2 = nn.Linear(2 * self.embed_d, self.embed_d)
        self.bd2 = nn.BatchNorm1d(self.embed_d + self.drug_dim + self.fpt_dim)
        self.bs2 = nn.BatchNorm1d(2 * self.embed_d)
        
        self.Fd = nn.Linear(self.drug_dim, self.dim)
        self.Fs = nn.Linear(self.embed_d, self.embed_d)
        self.Ft = nn.Linear(self.fpt_dim, self.dim)
        
        se_feature = nn.Parameter(torch.Tensor(self.S_n, self.embed_d))
        se_feature.data.normal_(0, 0.1)

        self.act = nn.LeakyReLU(args.alpha)

        self.drug_feature = mpnn_feature
        self.fpt_feature = fpt_feature
        self.se_feature = se_feature
        self.drug_se_train = drug_se_train
        self.se_drug_train = se_drug_train


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                nn.init.normal_(m.weight.data, std=0.1)


    def node_het_agg(self, node_type, drug_agg, se_agg):

        if node_type == 1:
            d_s_agg = torch.matmul(self.drug_se_train, se_agg)
            concate_embed = torch.cat((self.drug_feature, d_s_agg, self.fpt_feature), 1)
            concate_embed = self.act(self.Gd(self.bd(concate_embed)))
        elif node_type == 2:
            s_d_agg = torch.matmul(self.se_drug_train, drug_agg)
            concate_embed = torch.cat((self.se_feature, s_d_agg), 1)
            concate_embed = self.act(self.Gs(self.bs(concate_embed)))
        elif node_type == 3:
            d_s_agg = torch.matmul(self.drug_se_train, se_agg)
            concate_embed = torch.cat((self.drug_feature, d_s_agg, self.fpt_feature), 1)
            concate_embed = self.act(self.Gd2(self.bd2(concate_embed)))
        elif node_type == 4:
            s_d_agg = torch.matmul(self.se_drug_train, drug_agg)
            concate_embed = torch.cat((self.se_feature, s_d_agg), 1)
            concate_embed = self.act(self.Gs2(self.bs2(concate_embed)))

        return concate_embed

    def forward(self):
        drug_embedding = self.Fd(self.drug_feature)
        se_embedding = self.Fs(self.se_feature)
        fpt_embedding = self.Ft(self.fpt_feature)
        
        drug_embedding = torch.cat((drug_embedding, fpt_embedding), 1)
        
        drug_embedding = nn.functional.dropout(drug_embedding, self.dropout, training=self.training)
        se_embedding = nn.functional.dropout(se_embedding, self.dropout, training=self.training)

        drug_embeddings_t = self.node_het_agg(1, drug_embedding, se_embedding)
        se_embeddings_t = self.node_het_agg(2, drug_embedding, se_embedding)
        
        drug_embeddings_t = nn.functional.dropout(drug_embeddings_t, self.dropout, training=self.training)
        se_embeddings_t = nn.functional.dropout(se_embeddings_t, self.dropout, training=self.training)
        
        drug_embedding_e = self.node_het_agg(3, drug_embeddings_t, se_embeddings_t)
        se_embedding_e = self.node_het_agg(4, drug_embeddings_t, se_embeddings_t)
        
        drug_embedding_e = nn.functional.dropout(drug_embedding_e, self.dropout, training=self.training)
        se_embedding_e = nn.functional.dropout(se_embedding_e, self.dropout, training=self.training)
        
        drug_embeddings = self.Wd(self.bnd(drug_embedding_e))
        se_embeddings = self.Ws(self.bns(se_embedding_e))
        
        outputs = torch.mm(drug_embeddings, se_embeddings.t())
        return outputs
