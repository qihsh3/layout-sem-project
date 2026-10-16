"""Image metrics without dataset paths or embedded observations."""
import math
import numpy as np
from PIL import Image

def structural_similarity(x, y, data_range, win_size=7):
    """Mean SSIM on valid 7x7 windows, uniform weights, sample covariance."""
    from numpy.lib.stride_tricks import sliding_window_view
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape and x.ndim == 2
    xw = sliding_window_view(x, (win_size, win_size))
    yw = sliding_window_view(y, (win_size, win_size))
    mx, my = xw.mean((-2, -1)), yw.mean((-2, -1))
    n = win_size * win_size
    vx = ((xw - mx[..., None, None]) ** 2).sum((-2, -1)) / (n - 1)
    vy = ((yw - my[..., None, None]) ** 2).sum((-2, -1)) / (n - 1)
    cov = ((xw - mx[..., None, None]) * (yw - my[..., None, None])).sum((-2, -1)) / (n - 1)
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    return np.mean(((2 * mx * my + c1) * (2 * cov + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2)))

def metric(gt, pred, threshold=130):
    x = np.asarray(Image.fromarray(gt).convert('L'), dtype=float)
    y = np.asarray(Image.fromarray(pred).convert('L'), dtype=float)
    error = float(np.mean((x-y)**2))
    a, b = x > threshold, y > threshold
    union = (a | b).sum()
    return {'PSNR_gray': 10 * math.log10(255**2/error) if error else None,
            'SSIM_gray': float(structural_similarity(x, y, 255)),
            'SSIM_binary': float(structural_similarity(a.astype(float), b.astype(float), 1)),
            f'IoU_{threshold}': float((a & b).sum()/union) if union else 1.,
            'MAE_RGB': float(np.abs(gt.astype(float)-pred.astype(float)).mean())}

def means(records):
    return {k: float(np.mean([r[k] for r in records if r[k] is not None]))
            if any(r[k] is not None for r in records) else None for k in records[0]}
