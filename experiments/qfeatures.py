import itertools
from .common import *
import abc
from scipy import stats
from tqdm.auto import tqdm
def invertible(x:np.ndarray):
    n,m=x.shape
    assert n==m
    epsilon = np.finfo(x.dtype).eps
    condition_number = np.linalg.cond(x)
    return condition_number<1/epsilon

# from math import comb
import operator as op
from functools import reduce
def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

class DataReducedQ(StarExperiment):
    def description(self) -> str:
        return "Find reduced structure in Q features with blocks, using q features computed from a dataset"

    def run(self):
        
        coefficients_dict = datasets.aidelman.coefficients
        
        names, coefficients = list(coefficients_dict.keys()),list(coefficients_dict.values())
        
        for q in [3,4]: 
            # q_matrix,combinations = qfeatures.q_matrix_tied_sequential(coefficients,q=q)
            q_matrix,combinations = qfeatures.q_matrix_sequential_steps(coefficients,q=q)
            rows,columns = q_matrix.shape
            results = []
            n_combinations = comb(rows,columns)
            for combination_index_combinations in tqdm(itertools.combinations(range(rows),columns),total=n_combinations):
                q_matrix_reduced = q_matrix[combination_index_combinations,:]
                # is_invertible = invertible(q_matrix_reduced)
                # if is_invertible:
                #     total_correlation = np.triu(np.corrcoef(q_matrix_reduced),k=1)
                #     total_correlation = np.abs(total_correlation).sum()
                #     results.append((q_matrix_reduced,is_invertible,total_correlation,combination_index_combinations))

                #     print(combination_index_combinations,is_invertible,total_correlation)
            print(f"q = {q}")
            for r in results:
                print(r)

        
    def analyze_matrix(self,matrix:np.ndarray):
        n_features,n = matrix.shape
        epsilon = np.finfo(matrix.dtype).eps
        print(epsilon)
        sn.heatmap(matrix)
        self.save_close_fig(f"q{q}_n{n_features}_full")
        subsets_combinations = list(itertools.combinations(n_features,n_features))
        for i,subset_combinations in enumerate(subsets_combinations):
            subset_combination_matrix = generate_combination_matrix(subset_combinations,n_features)
            condition_number = np.linalg.cond(subset_combination_matrix)
            invertible = condition_number<1/epsilon
            invertible_str = "_sin" if not invertible else "_inv"
            sn.heatmap(subset_combination_matrix)
            subset_combinations_str = ",".join([str(s) for s in subset_combinations])
            plt.title(f"Invertible: {invertible}, condition number: {condition_number}\n{subset_combinations_str}")

            self.save_close_fig(f"q{q}_n{n_features}{invertible_str}_reduced{i:03}")

class ReducedQ(StarExperiment):
    def description(self) -> str:
        return "Find reduced structure in Q features with blocks"

    def run(self):
        
        def generate_combination_matrix(n_features,n_magnitudes):
            combination_matrix = np.zeros(n_features,n_magnitudes)
            for i,c in enumerate(itertools.combinations(n_features,n_magnitudes)):
                combination_matrix[i,c]=1
            return combination_matrix

        n_features = 5
        q = 3
        features = list(range(n_features))
        full_combinations = list(itertools.combinations(features,3))
        full_combination_matrix = generate_combination_matrix(full_combinations,n_features)
        

    def analyze_matrix(self,matrix:np.ndarray):
        n_features,n = matrix.shape
        epsilon = np.finfo(matrix.dtype).eps
        print(epsilon)
        sn.heatmap(matrix)
        self.save_close_fig(f"q{q}_n{n_features}_full")
        subsets_combinations = list(itertools.combinations(n_features,n_features))
        for i,subset_combinations in enumerate(subsets_combinations):
            subset_combination_matrix = generate_combination_matrix(subset_combinations,n_features)
            condition_number = np.linalg.cond(subset_combination_matrix)
            invertible = condition_number<1/epsilon
            invertible_str = "_sin" if not invertible else "_inv"
            sn.heatmap(subset_combination_matrix)
            subset_combinations_str = ",".join([str(s) for s in subset_combinations])
            plt.title(f"Invertible: {invertible}, condition number: {condition_number}\n{subset_combinations_str}")

            self.save_close_fig(f"q{q}_n{n_features}{invertible_str}_reduced{i:03}")
