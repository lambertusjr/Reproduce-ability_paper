def elliptic_pre_processing(classes_df, edgelist_df, features_df):
    #Geting known nodes
    known_nodes = classes_df[classes_df != 'unknown'].index
    #Remap class labels
    class_mapping = {
        'unknown': 0,
        '1': 1,
        '2': 2
    }
    classes_df['class'] = classes_df['class'].map(class_mapping)
    return classes_df, edgelist_df, features_df, known_nodes
import torch
from torch_geometric.data import Data
def create_data_object(features_df, edgelist_df, classes_df, node_mask = None, edge_mask = None):
    #Converting dataframes to tensors
    features_tensor = torch.tensor(features_df.deop(columns=['txId']).values, dtype=torch.float)
    edgelist_tensor = torch.tensor(edgelist_df.values.T, dtype=torch.long)
    classes_tensor = torch.tensor(classes_df['class'].values, dtype=torch.long)
    
    data = Data(x = features_tensor, edge_index=edgelist_tensor, y=classes_tensor)
    
    return data