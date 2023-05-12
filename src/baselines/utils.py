import numpy as np
from hierarchicalforecast.evaluation import scaled_crps, msse
from statistics import mean, stdev
import torch as th

def lag_dataset(seqs, back):
    X, Y = [], []
    for i in range(back, seqs.shape[1]):
        X.append(seqs[:, :i-1])
        Y.append(seqs[:, i])
    return X, Y

def get_hmatrix(aggmatrix):
    n_series = len(aggmatrix)
    n_bottom = len(aggmatrix[0])
    hmatrix = np.zeros((n_series, n_series))
    for r in range(n_series):
        if sum(aggmatrix[r]) == 0:
            continue
        idx_set = set()
        for c in range(n_bottom):
            if aggmatrix[r][c] == 1.0:
                for i in range(r+1, n_series):
                    if aggmatrix[i][c] == 1.0:
                        idx_set.add(i)
                        break
        if len(idx_set) == 0:
            idx_set.add(r)
        for i in idx_set:
            hmatrix[r][i] = 1.0
    return hmatrix

def jsd_norm(mu1, mu2, var1, var2):
    mu_diff = mu1 - mu2
    t1 = 0.5 * (mu_diff ** 2 + (var1) ** 2) / (2 * (var2) ** 2)
    t2 = 0.5 * (mu_diff ** 2 + (var2) ** 2) / (2 * (var1) ** 2)
    return t1 + t2 - 1.0

def jsd_loss(mu, logstd, hmatrix, train_means, train_std):
    eps = 0.0001
    lhs_mu = (((mu * train_std + train_means) * hmatrix).sum(1) - train_means) / (
        train_std
    )
    lhs_var = (((th.exp(2.0 * logstd) * (train_std ** 2)) * hmatrix).sum(1)) / (
        train_std ** 2
    )
    ans = th.nan_to_num(jsd_norm(mu, lhs_mu, (2.0 * logstd).exp(), lhs_var+eps))
    return ans.mean()

def calc_bootstrap(result_list):
    # Initialize an empty list to store the result
    result = []

    # Iterate over the indices of the elements in the sublists
    for i in range(len(result_list[0])):
        # Get the elements at the same position from all the sublists
        elements = [sublist[i] for sublist in result_list]
        # Calculate the mean and standard deviation of the elements
        mean_val = mean(elements)
        std_val = stdev(elements)
        # Append the formatted string to the result list
        result.append(f'{mean_val:.4f}±{(1.96 * std_val):.4f}')
    
    return result