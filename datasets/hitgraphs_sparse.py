"""Dataset specification for hit graphs using pytorch_geometric formulation"""

# System imports
import os

# External imports
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, ConcatDataset
import torch_geometric
from torch_geometric.data import Batch

LOAD_BALANCE=False

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

class HitGraphDataset(ConcatDataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir, n_samples=None, real_weight=1.0):
        input_dir = os.path.expandvars(input_dir)
        filenames = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                     if f.startswith('event') and not f.endswith('_ID.npz')]
        self.filenames = filenames if n_samples is None else filenames[:n_samples]
        self.filenames_sorted = filenames if n_samples is None else filenames[:n_samples]
        self.real_weight = real_weight
        self.fake_weight = real_weight / (2 * real_weight - 1)
        self._sorted_indices = []
        self._sorted_indices = list(range(len(self)))
        #self._batch = 
        #self.num_edge_index = []
        

    def __getitem__(self, index):

        if(type(index)==list):
            mybatch = []
            for idx in index:
                x, edge_index, y = load_graph(self.filenames[idx])
                w = y * self.real_weight + (1-y) * self.fake_weight
                mybatch.append(torch_geometric.data.Data(x=torch.from_numpy(x),
                                         edge_index=torch.from_numpy(edge_index),
                                         y=torch.from_numpy(y), w=torch.from_numpy(w)))
#            self._batch = batch
            print("WARNING USING LIST WITH __GETITEM__")
            return Batch.from_data_list(mybatch)
            
                

        else:
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
        self._sorted_indices = index_list 
        


    def __len__(self):
        return len(self.filenames)

def get_datasets(input_dir, n_train, n_valid, real_weight=1.0):
    
    data = HitGraphDataset(input_dir, n_train + n_valid, real_weight=real_weight)
    #data = 
    
    #print("worker info: ", torch.utils.data.get_worker_info())

    # Split into train and validation
    train_data, valid_data = random_split(data, [n_train, n_valid])
    if(LOAD_BALANCE):
        # Sort data by Number of Edges
        data_items = [e for e in train_data]
        #data.num_edges = data.num_edges
        print("length of data_items: ", len(data_items))
        print("last example number of edge features: ",  data_items[-1].dataset.num_edges)
        print("last example number of node features: ",  data_items[-1].dataset.num_nodes)
        edge_sizes_by_idx = np.array([ (i, e.num_edges) for i,e in enumerate(data_items.dataset)],dtype=np.int64)
        edges_sorted = np.sort(edge_sizes_by_idx,axis=0)
        print("last ten items in edges_sorted: ", edges_sorted[-10:-1])
        #sorted_idxes = np.array([int(e) for e in edges_sorted[:,0]],dtype=int)
        #sorted_idxes.reshape((1,-1))
        # Split into train and validation
        #train_data, valid_data = random_split(data, [n_train, n_valid])
        #print("original train_data:",train_data[-1])
        sorted_edge_idx = [int(e[0]) for e in edges_sorted]

        train_data.dataset.reorder_items(sorted_edge_idx)
        

        # Split into train and validation
        #train_data, valid_data = random_split(data, [n_train, n_valid])
        #offline_dataset._sorted_indices
        #train_data.dataset._sorted_indices = sorted_edge_idx 
        
        print("sorted train_data:", train_data)
        #train_data = train_data[sorted_edge_idx]
    #else:
        # Split into train and validation
    #    train_data, valid_data = random_split(data, [n_train, n_valid])

    return train_data, valid_data
