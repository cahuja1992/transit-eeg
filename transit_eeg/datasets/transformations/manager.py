from transit_eeg.datasets.transformations.registry import TRANSFORMATIONS

class TransformManager:
    def __init__(self, transform_cfg):
        self.transforms = [TRANSFORMATIONS.build(cfg) for cfg in transform_cfg]

    def apply(self, data):
        for t in self.transforms:
            data = t.transform(data)
        return data

    def initialize(self, data, meta=None, reshape=None, is_feature=True):
        if reshape is not None:
            data = data.reshape(-1, *reshape)
        shape = list(data.shape)
        pre_fft = True
        for t in self.transforms:
            if hasattr(t, "init_from_meta"):
                t.init_from_meta(meta, is_feature, pre_fft)
            if hasattr(t, "initialize"):
                t.initialize(data)
            if hasattr(t, "shape_transform"):
                shape = t.shape_transform(shape)
            if t.__class__.__name__.lower().startswith("fft"):
                pre_fft = False
        return shape

    def __repr__(self):
        return f"TransformManager: {', '.join([str(t) for t in self.transforms])}"