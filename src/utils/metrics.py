import numpy as np


def HR(pos_matrix, topk):
    """
    :param pos_matrix: (-1, max(topk)) shape
    :param topk: top-K value list
    :return: hit ratio
    """
    hit = np.cumsum(pos_matrix, axis=1)
    res = [hit[:, k-1] for k in topk]
    return res

def PRECISION(pos_matrix, topk):
    """
    :param pos_matrix: (-1, max(topk)) shape
    :param topk: top-K value list
    :return: precision
    """
    hit = np.cumsum(pos_matrix, axis=1)
    res = [hit[:, k-1] / k for k in topk]
    return res

def MAP(pos_matrix, topk):
    precision = np.cumsum(pos_matrix, axis=1) / np.arange(1, pos_matrix.shape[1] + 1)
    sum_pre = np.cumsum(precision * pos_matrix.astype(np.float32), axis=1)
    ranges = np.arange(1, pos_matrix.shape[1] + 1)
    res = [sum_pre[:, k-1]/ranges[k-1] for k in topk]
    return res

def MRR(pos_matrix, topk):
    """
    :param pos_matrix: (-1, max(topk)) shape
    :param topk: top-K value list
    :return: mrr
    """
    idx = np.argmax(pos_matrix, axis=1)
    mrr = np.zeros_like(pos_matrix, dtype=np.float32)
    for row, idx in enumerate(idx):
        if pos_matrix[row, idx] > 0:
            mrr[row, idx:] = 1 / (idx + 1)
        else:
            mrr[row, idx:] = 0
    res = [mrr[:, k-1] for k in topk]
    return res

def NDCG(pos_matrix, topk):
    """
    :param pos_matrix: (-1, max(topk)) shape
    :param topk: top-K value list
    :return: ndcg
    """
    idx = np.arange(1, pos_matrix.shape[1] + 1)
    dcg = np.cumsum(pos_matrix / np.log2(idx + 1), axis=1)
    idcg = 1.0 # only one positive item in the ground-truth
    ndcg = dcg / idcg
    res = [ndcg[:, k-1] for k in topk]
    return res
