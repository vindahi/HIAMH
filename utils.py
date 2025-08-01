import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch
import logging
import os.path as osp
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel


def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def zero2eps(x):

    x[x == 0] = 1
    return x

def compute_centers(x, psedo_labels, num_cluster):
    n_samples = x.size(0)
    if len(psedo_labels.size()) > 1:
        weight = psedo_labels.T
    else:
        weight = torch.zeros(num_cluster, n_samples).to(x)  # L, N
        weight[psedo_labels, torch.arange(n_samples)] = 1

    weight = weight.float()
    weight = F.normalize(weight, p=1, dim=1) 
    centers = torch.mm(weight, x)
    centers = F.normalize(centers, dim=1)
    return centers




def compute_cluster_loss(q_centers, k_centers, temperature, psedo_labels, num_cluster):
    d_q = q_centers.mm(q_centers.T) / temperature
    d_k = (q_centers * k_centers).sum(dim=1) / temperature
    d_q = d_q.float()
    d_q[torch.arange(num_cluster), torch.arange(num_cluster)] = d_k
    zero_classes = torch.nonzero(torch.sum(psedo_labels, dim=0) == 0).squeeze()
    mask = torch.zeros((num_cluster, num_cluster), dtype=torch.bool, device=d_q.device)
    mask[:, zero_classes] = 1
    d_q.masked_fill_(mask, -10)

    pos = d_q.diag(0)
    mask = torch.ones((num_cluster, num_cluster))
    mask = mask.fill_diagonal_(0).bool()

    neg = d_q[mask].reshape(-1, num_cluster - 1)
    loss = - pos + torch.logsumexp(torch.cat([pos.reshape(num_cluster, 1), neg], dim=1), dim=1)
    loss[zero_classes] = 0.
    if zero_classes.numel() < num_cluster:
        loss = loss.sum() / (num_cluster - zero_classes.numel())
    else:
        loss = 0.

    return loss


def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum
    in_affnty = np.transpose(affinity/row_sum)
    return in_affnty, out_affnty


def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    in_aff, out_aff = normalize(affinity_matrix)
    return in_aff, out_aff, affinity_matrix





def calculate_map(qu_B, re_B, qu_L, re_L):
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap



def calculate_hamming(B1, B2):
    leng = B2.shape[1] 
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def qmi_loss(code, targets, eps=1e-8):

    code = code / (torch.sqrt(torch.sum(code ** 2, dim=1, keepdim=True)) + eps)
    Y = torch.mm(code, code.t())
    Y = 0.5 * (Y + 1)
    targets = targets.float()
    D = targets.mm(targets.transpose(0, 1)) > 0
    D = D.type(torch.cuda.FloatTensor)

    M = D.size(1) ** 2 / torch.sum(D)

    Qy_in = (D * Y - 1) ** 2
    Qy_btw = (1.0 / M) * Y ** 2
    loss = torch.sum(Qy_in + Qy_btw)
    return loss





def calculate_pr_curve(qB, rB, query_label, retrieval_label):
    # 确保输入为 PyTorch 张量
    if isinstance(query_label, np.ndarray):
        query_label = torch.Tensor(query_label)
    if isinstance(retrieval_label, np.ndarray):
        retrieval_label = torch.Tensor(retrieval_label)
    if isinstance(qB, np.ndarray):
        qB = torch.Tensor(qB)
    if isinstance(rB, np.ndarray):
        rB = torch.Tensor(rB)

    num_query = query_label.shape[0]
    all_precision = []
    all_recall = []

    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd).item()
        
        if tsum == 0:
            continue
        
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        sorted_indices = torch.argsort(hamm)

        tp = 0
        fp = 0
        precision = []
        recall = []

        for index in sorted_indices:
            if gnd[index] == 1:  # True Positive
                tp += 1
            else:  # False Positive
                fp += 1
            
            # Compute precision and recall
            if tp + fp > 0:
                precision_value = tp / (tp + fp)
            else:
                precision_value = 0.0
            
            recall_value = tp / tsum
            
            precision.append(precision_value)
            recall.append(recall_value)

        all_precision.append(precision)
        all_recall.append(recall)

    # Average over all queries
    avg_precision = [sum(p) / num_query for p in zip(*all_precision)]
    avg_recall = [sum(r) / num_query for r in zip(*all_recall)]

    return avg_precision, avg_recall





def p_topK(qB, rB, query_label, retrieval_label, K=None):
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p

def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def p_topK2(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        hamm = torch.Tensor(hamm)
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm).indices[:int(total)]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p

def get_rank(model):

    #### If our paper is accepted, the key code will be released

    return 0



def center_loss(centroids):
    centroids_dist = torch.cdist(centroids, centroids, p=2) / centroids.shape[1]
    triu_dist = torch.triu(centroids_dist, diagonal=1)

    non_zero_dist = triu_dist[triu_dist > 0]
    mean_dist = torch.mean(non_zero_dist) if non_zero_dist.numel() > 0 else 0

    min_dist = torch.min(triu_dist[triu_dist > 0]) if non_zero_dist.numel() > 0 else 0

    reg_term = mean_dist + min_dist
    return reg_term


def logger(fileName='log'):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    log_name = str(fileName) + '.log'
    log_dir = './logs'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)
    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)
    return logger
