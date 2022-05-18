
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from tensorflow import keras
import datasets
import numpy as np
import pandas as pd

import seaborn as sn
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 300

from .base import BeExperiment,StarExperiment,Magnitude,QFeature,Feature
import qfeatures
from preprocessing import outliers

from .models import *


common_features = [Magnitude(),QFeature(3),QFeature(4)]
