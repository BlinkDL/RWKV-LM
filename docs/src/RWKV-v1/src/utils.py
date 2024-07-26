import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

def top_p_probs(probs, p):
    out = probs.clone()

    sorted_probs, sorted_indices = torch.sort(out, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0    
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    out[indices_to_remove] = 0

    return out

# top-p + top-k + pow&ratio sampling
def sample_logits(logits, pos, temperature=1.0, top_k=None, top_p=None, min_p_pow=None, min_p_ratio=None):
    logits = logits[:, pos, :] / temperature
    probs = F.softmax(logits, dim=-1)
    
    if min_p_ratio is not None:
        limit = torch.pow(torch.max(probs), min_p_pow) * min_p_ratio
        logits[probs < limit] = -float('Inf')
    
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    
    probs = F.softmax(logits, dim=-1)
    
    if top_p is not None:
        probs[0] = top_p_probs(probs[0], top_p)
    
    ix = torch.multinomial(probs, num_samples=1)
    return ix[0][0].cpu()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
