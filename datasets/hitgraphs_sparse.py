"""Dataset specification for hit graphs using pytorch_geometric formulation"""

# System imports
import os

# External imports
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import torch_geometric
LOAD_BALANCE=True

def load_graph(filename):
    with np.load(filename) as f:
        x, y = f['X'], f['y']
        Ri_rows, Ri_cols = f['Ri_rows'], f['Ri_cols']
        Ro_rows, Ro_cols = f['Ro_rows'], f['Ro_cols']
        n_edges = Ri_cols.shape[0]
        edge_index = np.zeros((2, n_edges), dtype=int)
        edge_index[0, Ro_cols] = Ro_rows
        edge_index[1, Ri_cols] = Ri_rows
    return x, edge_index, y

class HitGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir, n_samples=None, real_weight=1.0):
        input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if f.startswith('event') and not f.endswith('_ID.npz')]
        self.filenames = filenames if n_samples is None else filenames[:n_samples]
        self.filenames_sorted = filenames if n_samples is None else filenames[:n_samples]
        self.real_weight = real_weight
        self.fake_weight = real_weight / (2 * real_weight - 1)
        #self.num_edge_index = []

    def __getitem__(self, index):

        x, edge_index, y = load_graph(self.filenames[index])
        # Compute weights
        w = y * self.real_weight + (1-y) * self.fake_weight
        #self.num_edge_index.append( [edge_index[0].shape[0],index] )
        
        #print("num_edges = ", edge_index[0].shape[0], "for index=",index )
        return torch_geometric.data.Data(x=torch.from_numpy(x),
                                         edge_index=torch.from_numpy(edge_index),
                                         y=torch.from_numpy(y), w=torch.from_numpy(w))

    def reorder_items(self, index_list):

        for i,index in enumerate(index_list):        
            self.filenames_sorted[i] = self.filenames[index]

        print("Successfully reassigned file list according to sorted index array")
        self.filenames = self.filenames_sorted 
        


    def __len__(self):
        return len(self.filenames)

def get_datasets(input_dir, n_train, n_valid, real_weight=1.0):
    
    data = HitGraphDataset(input_dir, n_train + n_valid, real_weight=real_weight)
    #print("worker info: ", torch.utils.data.get_worker_info())

    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])
    
    # Sort data by Number of Edges
    data_items = [e for e in data]
    #data.num_edges = data.num_edges
    print("length of data_items: ", len(data_items))
    print("last example number of edge features: ",  data_items[-1].num_edges)
    print("last example number of node features: ",  data_items[-1].num_nodes)
    edge_sizes_by_idx = np.array([ (i, e.num_edges) for i,e in enumerate(data_items)],dtype=np.int64)
    edges_sorted = np.sort(edge_sizes_by_idx,axis=0)
    print("last ten items in edges_sorted: ", edges_sorted[-10:-1])
    #sorted_idxes = np.array([int(e) for e in edges_sorted[:,0]],dtype=int)
    #sorted_idxes.reshape((1,-1))


    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])

    print("original train_data:",train_data[-1])
    sorted_edge_idx = [int(e[0]) for e in edges_sorted]
    data.reorder_items(sorted_edge_idx)
    print("sorted train_data:", train_data[-1])
    #train_data = train_data[sorted_edge_idx]
    return train_data, valid_data
