
from facu import preprocess
import numpy as np
data,class_names=preprocess.load_data()
x,y,id = data
import matplotlib.pyplot as plt

def plot_class_distribution(y:np.ndarray,filepath:str):
    unique, counts = np.unique(y, return_counts=True)
    print(unique,counts)
    plt.bar(unique,counts)
    plt.savefig(filepath)
    plt.close()


plot_class_distribution(y,"plots/class_histogram.png")




