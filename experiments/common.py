
import datasets
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from .base import Experiment
import qfeatures
from preprocessing import outliers


class StarExperiment(Experiment):


    def save_close_fig(self,filename):
        plt.tight_layout()
        plt.savefig(self.plot_folderpath / filename)
        plt.close()