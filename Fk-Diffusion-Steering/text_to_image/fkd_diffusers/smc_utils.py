import numpy as np
import torch as th
from numpy import random

try:
    from numba import jit
except Exception:
    # Fallback: run without numba JIT when numba is unavailable.
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


def compute_ess(w, dim=-1):
    ess = (w.sum(dim=dim)) ** 2 / th.sum(w**2, dim=dim)
    return ess


def normalize_log_weights(log_weights, dim):
    log_weights = log_weights - log_weights.max(dim=dim, keepdims=True)[0]
    log_weights = log_weights - th.logsumexp(log_weights, dim=dim, keepdims=True)
    return log_weights


@jit(nopython=True)
def inverse_cdf(su, W):
    j = 0
    s = W[0]
    M = su.shape[0]
    A = np.empty(M, dtype=np.int64)
    for n in range(M):
        while su[n] > s:
            if j == M - 1:
                break
            j += 1
            s += W[j]
        A[n] = j
    return A


def uniform_spacings(N):
    z = np.cumsum(-np.log(random.rand(N + 1)))
    return z[:-1] / z[-1]


def multinomial(W, M):
    return inverse_cdf(uniform_spacings(M), W)


def stratified(W, M):
    su = (random.rand(M) + np.arange(M)) / M
    return inverse_cdf(su, W)


def systematic(W, M):
    su = (random.rand(1) + np.arange(M)) / M
    return inverse_cdf(su, W)


def residual(W, M):
    N = W.shape[0]
    A = np.empty(M, dtype=np.int64)
    MW = M * W
    intpart = np.floor(MW).astype(np.int64)
    sip = np.sum(intpart)
    res = MW - intpart
    sres = M - sip
    A[:sip] = np.arange(N).repeat(intpart)
    if sres > 0:
        A[sip:] = multinomial(res / sres, M=sres)
    return A


@jit(nopython=True)
def ssp(W, M):
    N = W.shape[0]
    MW = M * W
    nr_children = np.floor(MW).astype(np.int64)
    xi = MW - nr_children
    u = random.rand(N - 1)
    i, j = 0, 1
    for k in range(N - 1):
        delta_i = min(xi[j], 1.0 - xi[i])
        delta_j = min(xi[i], 1.0 - xi[j])
        sum_delta = delta_i + delta_j
        pj = delta_i / sum_delta if sum_delta > 0.0 else 0.0
        if u[k] < pj:
            j, i = i, j
            delta_i = delta_j
        if xi[j] < 1.0 - xi[i]:
            xi[i] += delta_i
            j = k + 2
        else:
            xi[j] -= delta_i
            nr_children[i] += 1
            i = k + 2
    if np.sum(nr_children) == M - 1:
        last_ij = i if j == k + 2 else j
        if xi[last_ij] > 0.99:
            nr_children[last_ij] += 1
    if np.sum(nr_children) != M:
        raise ValueError("ssp resampling: wrong size for output")
    return np.arange(N).repeat(nr_children)


def treeg(W, M):
    K = M // 2
    topk = np.argsort(W)[-K:]
    return np.repeat(topk, 2)


Resample_dict = dict(
    systematic=systematic,
    stratified=stratified,
    residual=residual,
    multinomial=multinomial,
    ssp=ssp,
    treeg=treeg,
)


def resampling_function(resample_strategy="systematic", ess_threshold=None, verbose=False):
    resample_fn = Resample_dict[resample_strategy]

    def resample(log_w):
        assert log_w.dim() == 2, "Dimension of log_w should be 2"

        log_normalized_weights = normalize_log_weights(log_w, dim=-1)
        normalized_weights = th.exp(log_normalized_weights)
        P = log_w.shape[-1]
        ess = compute_ess(normalized_weights, dim=-1)

        resample_indices = th.zeros_like(log_w, device=log_w.device, dtype=th.int)
        is_resampled = th.zeros(log_w.shape[0], device=log_w.device, dtype=th.bool)

        for i, ess_batch in enumerate(ess):
            if ess_threshold is None or ess_batch < P * ess_threshold:
                if verbose:
                    print("resample")
                resample_indices[i] = th.from_numpy(resample_fn(W=np.array(normalized_weights[i].cpu()), M=P))
                is_resampled[i] = True
                log_w[i] = -th.log(th.Tensor([P]).to(log_w.device))
            else:
                resample_indices[i] = th.arange(P)
                is_resampled[i] = False
                log_w[i] = log_normalized_weights[i]

        return resample_indices, is_resampled, log_w

    return resample
