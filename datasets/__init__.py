"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader 
#from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from torch.utils.data.dataloader import default_collate


def get_data_loaders(name, batch_size, distributed=False,
                     n_workers=0, rank=None, n_ranks=None, load_balance=None,
                     **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate
    if name == 'dummy':
        from .dummy import get_datasets
        train_dataset, valid_dataset = get_datasets(**data_args)
    elif name == 'hitgraphs':
        from . import hitgraphs
        train_dataset, valid_dataset = hitgraphs.get_datasets(**data_args)
        collate_fn = hitgraphs.collate_fn
    elif name == 'hitgraphs_sparse':
        from torch_geometric.data import Batch
        from . import hitgraphs_sparse
        #from .loadbal_sampler import Batch
        if load_balance==True:
            print("Building Load-Balanced datasets")
            train_dataset, valid_dataset = hitgraphs_sparse.get_datasets(**data_args, load_balance=True)
        else:
            print("Building original datasets")
            train_dataset, valid_dataset = hitgraphs_sparse.get_datasets(**data_args, load_balance=False)

        collate_fn = Batch.from_data_list 
    else:
        raise Exception('Dataset %s unknown' % name)

    # Construct the data loaders
    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
    train_sampler, valid_sampler = None, None
    if distributed:

        if load_balance==True:
            print("Using Load-Balancing Bucket Schedule DataLoader and Sampling Scheme")
        #train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
        #valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
        #def __init__(self, offline_dataset, batch_size, shuffle_batches=True, num_buckets=None, shuffle_buckets=True):
            from .loadbal_sampler import DistributedGraphBatchSampler 
            train_sampler = DistributedGraphBatchSampler(train_dataset, batch_size, shuffle_batches=False, num_buckets=4, shuffle_buckets=False, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedGraphBatchSampler(valid_dataset, batch_size, shuffle_batches=False, num_buckets=4, shuffle_buckets=False, rank=rank, num_replicas=n_ranks)

        
            train_data_loader = DataLoader(train_dataset.dataset, sampler=train_sampler,
                                   shuffle=False, **loader_args)
                                   #shuffle=(train_sampler is None), **loader_args)
            valid_data_loader = (DataLoader(valid_dataset.dataset, sampler=valid_sampler, **loader_args)
                         if valid_dataset is not None else None)

        else:

            print("Using Original DataLoader and Sampling Scheme")
            train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=n_ranks)
            valid_sampler = DistributedSampler(valid_dataset, rank=rank, num_replicas=n_ranks)
  
    
            train_data_loader = DataLoader(train_dataset.dataset, sampler=train_sampler,
                                   shuffle=(train_sampler is None), **loader_args)
            valid_data_loader = (DataLoader(valid_dataset.dataset, sampler=valid_sampler, **loader_args)
                                     if valid_dataset is not None else None)


    return train_data_loader, valid_data_loader
