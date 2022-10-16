class BaseMetric:
    def __init__(self, name=None, *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.use_bs_pred = False

    def __call__(self, **batch):
        raise NotImplementedError()
