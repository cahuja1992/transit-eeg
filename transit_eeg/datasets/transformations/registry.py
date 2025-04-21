class Registry(dict):
    def register(self, name=None):
        def decorator(fn_or_cls):
            key = name or fn_or_cls.__name__.lower()
            self[key] = fn_or_cls
            return fn_or_cls
        return decorator

    def build(self, cfg):
        assert "type" in cfg
        cls = self[cfg["type"]]
        params = {k: v for k, v in cfg.items() if k != "type"}
        return cls(**params)

TRANSFORMATIONS = Registry()