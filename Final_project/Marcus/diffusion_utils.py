import math
import torch
import torch.nn as nn

def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL( N(mean1, exp(logvar1)) || N(mean2, exp(logvar2)) ), per sample in the batch.
    mean1, logvar1, mean2, logvar2: (batch_size, D)
    Returns: (batch_size,)
    """
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    D = mean1.shape[1]

    # log_det_ratio = sum(logvar2 - logvar1)
    log_det_ratio = (logvar2 - logvar1).sum(dim=1)
    # trace term: sum( var1 + (mean1-mean2)^2 ) / var2
    diff = (mean1 - mean2)**2
    trace = (var1 + diff) / var2
    return 0.5 * (log_det_ratio + trace.sum(dim=1) - D)

def gaussian_nll(x, mean, logvar):
    """
    Negative log-likelihood of x under N(mean, exp(logvar)).
    x, mean, logvar: (batch_size, D)
    Returns: (batch_size,)
    """
    var = torch.exp(logvar)
    D = x.shape[1]
    log_det = logvar.sum(dim=1)  # sum of log-variances
    diff = (x - mean)**2
    mahal = (diff / var).sum(dim=1)
    return 0.5 * (D*math.log(2*math.pi) + log_det + mahal)

def pred_xstart_from_eps(x_t, t, eps_hat, alpha_bar):
    """
    x_t:     (B, D)
    t:       (B,) or scalar
    eps_hat: (B, D)   network’s predicted noise
    alpha_bar: precomputed alpha_bar[t]  of shape (B,) or broadcastable

    returns: (B, D)  predicted x0
    """
    # if alpha_bar is shape (B,), expand to (B,1) to multiply x_t/eps in a batchwise way
    if alpha_bar.dim() == 0:
        # alpha_bar is scalar
        sqrt_ab   = torch.sqrt(alpha_bar).view(1,1)     # or .unsqueeze(0).unsqueeze(1)
        sqrt_1mab = torch.sqrt(1 - alpha_bar).view(1,1)
    else:
        # alpha_bar is (B,) already
        sqrt_ab   = torch.sqrt(alpha_bar).unsqueeze(1)     # (B,1)
        sqrt_1mab = torch.sqrt(1 - alpha_bar).unsqueeze(1)

    return (x_t - sqrt_1mab * eps_hat) / sqrt_ab
