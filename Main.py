# Unknown = Unknown
# 1 = Illicit
# 2 = Licit
#Code for paper reproduce-ablility
seeded_run = True
#Detecting system (pc or mac)
import platform
pc = platform.system()

import os
if pc == 'Darwin':
    os.chdir("/Users/lambertusvanzyl/Desktop/Reproduce-ability_paper")
    #/Users/lambertusvanzyl/Desktop/Reproduce-ability_paper
else:
    os.chdir("/Users/Lambertus/Desktop/Reproduce-ablility_paper")
    #"C:\Users\Lambertus\Desktop\AIO_GNN"
    
#%% Importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import networkx as nx

from functools import partial

#Importing PyTorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch import Tensor
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool

#Importing sklearn libraries
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import sys

#%% Importing custom libraries
from reading_data import readfiles
from pre_processing import elliptic_pre_processing, create_data_object, create_elliptic_masks

#%% Setting seed
if seeded_run == True:
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    
#%%Getting data ready for models
#Reading in data
features_df, classes_df, edgelist_df = readfiles(pc)
classes_df, edgelist_df, features_df, known_nodes = elliptic_pre_processing(classes_df, edgelist_df, features_df)
# Create data object
data = create_data_object(features_df, edgelist_df, classes_df)
#Create mask for data
create_elliptic_masks(features_df, edgelist_df, known_nodes)

#%% Testing if model runs
