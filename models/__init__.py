"""
Python module for holding our PyTorch models.
"""

def get_model(name, **model_args):
    """
    Top-level factory function for getting your models.
    """
    if name == 'cnn_classifier':
        from .cnn_classifier import CNNClassifier
        return CNNClassifier(**model_args)
    elif name == 'gnn_segment_classifier':
        from .gnn import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'gnn_sparse':
        from .gnn_sparse import GNNSegmentClassifier
        return GNNSegmentClassifier(**model_args)
    elif name == 'mpnn':
        from .mpnn import GNN
        return GNN(**model_args)
    else:
        raise Exception('Model %s unknown' % name)
