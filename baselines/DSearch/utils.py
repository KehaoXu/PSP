import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_metrics(r_batch):
    return_dict = {}
    for key in r_batch:
        value = r_batch[key]
        if key == 'diversity':
            value = torch.concat(value, dim=0)  # bs * dim
            # value = F.normalize(value, p=2, dim=1)
            cosine_matrix = cosine_similarity(value)
            # cosine_matrix = torch.mm(value, value.T)

            bs = cosine_matrix.shape[0]
            cosine_matrix_no_diag = cosine_matrix - np.eye(bs)
            cosine_similarity_matrix = cosine_matrix_no_diag.sum() / (bs * (bs - 1) + 1e-8)
            mean_value = 1 - cosine_similarity_matrix.item()
        else:
            mean_value = np.mean(value).item()
        return_dict[f'{key}'] = mean_value
    return return_dict
