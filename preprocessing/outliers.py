from scipy import stats
import numpy as np

def detect_outliers_iqr(x,iqr_factor= 3):

    q25,q75=x.quantile(0.25),x.quantile(0.75)
    iqr=q75-q25
    min_values = q25-iqr_factor*iqr
    max_values = q75+iqr_factor*iqr
    # ou
    indices = (np.logical_or(x<min_values,x>max_values)).any(axis=1)
    return indices


    
def detect_outliers_confidence_interval(x,confidence= 0.999,verbose=False):
    m = len(x.columns) # number of columns = number of hypothesis

    adjusted_confidence = 1- (1-confidence)/m  # bonferroni-adjusted confidence 
    max_zscore = stats.norm.ppf(adjusted_confidence)
    if verbose:
        print(f"Confidence  (desired): {confidence}")
        print(f"Confidence (adjusted): {adjusted_confidence}")
        print(f"Z-score    (adjusted): {max_zscore}")

    indices = (np.abs(stats.zscore(x-x.mean())) > max_zscore).any(axis=1)
    return indices