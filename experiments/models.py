from .models_configs import *


class SmallNNConfig(KerasDenseConfig):
    def __init__(self): super().__init__([64,32,16, 8])

class MediumNNConfig(KerasDenseConfig):
    def __init__(self): super().__init__([256,128,64,32,16])

class LargeNNConfig(KerasDenseConfig):
    def __init__(self): super().__init__([1024,512,256,64,32])

class LargeRandomForestConfig(RandomForestConfig):
    def __init__(self): super().__init__(10)

class LargeGradientBoostingConfig(GradientBoostingConfig):
    def __init__(self): super().__init__(6,300,0.9)    