#Code for paper reproduce-ablility

#Detecting system (pc or mac)
import platform
pc = platform.system()

import os
if pc == 'Darwin':
    os.chdir("/Users/lambertusvanzyl/Desktop/Reproduce-ablility_paper")
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
from sklearn.metrics import precisio, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import sys

#%% Importing custom libraries


#%% Setting seed
