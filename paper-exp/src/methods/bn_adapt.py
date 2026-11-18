from src.methods.base import Adapter, configure_norm


class BNAdapt(Adapter):
    timing = "current-batch-statistics"

    def __init__(self, model):
        super().__init__(model)
        configure_norm(self.model)
