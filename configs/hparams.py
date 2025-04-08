def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(dataset_name))
    return globals()[dataset_name]


class supervised():
    def __init__(self):
        super(supervised, self).__init__()
        self.train_params = {
            'num_epochs': 200,
            'batch_size': 64,
            'weight_decay': 1e-4,
            'learning_rate': 1e-4,
            'feature_dim': 1*128
        }