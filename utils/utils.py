import numpy as np
import math
import torch

def cxy_wh_2_rect(pos, sz):
    return [float(max(float(0), pos[0] - sz[0] / 2)), float(max(float(0), pos[1] - sz[1] / 2)), float(sz[0]),
            float(sz[1])]  # 0-index


def hann1d(sz: int, centered = True) -> np.ndarray:
    """1D cosine window."""
    if centered:
        return 0.5 * (1 - np.cos((2 * math.pi / (sz + 1)) * np.arange(1, sz + 1).astype(np.float64)))
    w = 0.5 * (1 + np.cos((2 * math.pi / (sz + 2)) * np.arange(0, sz//2 + 1).astype(np.float64)))
    return np.concatenate([w, np.flip(w[1:sz-sz//2], (0,))])


def hann2d(sz: np.ndarray, centered = True) -> np.ndarray:
    """2D cosine window."""
    return hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)
    
# def hann1d(sz: int, centered = True) -> np.ndarray:
#     """1D cosine window."""
#     if centered:
#         return 0.5 * (1 - torch.cos((2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
#     w = 0.5 * (1 + torch.cos((2 * math.pi / (sz + 2)) * torch.arange(0, sz//2 + 1).float()))
#     return torch.cat([w, w[1:sz-sz//2].flip((0,))]).numpy()


# def hann2d(sz: np.ndarray, centered = True) -> np.ndarray:
#     """2D cosine window."""
#     sz = torch.from_numpy(sz)
#     ret = hann1d(sz[0].item(), centered).reshape(1, 1, -1, 1) * hann1d(sz[1].item(), centered).reshape(1, 1, 1, -1)
#     return ret.numpy()

def img2tensor(img):
    img = img[..., ::-1]  # BGR2RGB
    img = img - (0.485*255, 0.456*255, 0.406*255)
    img = img * (1/(0.229*255), 1/(0.224*255), 1/(0.225*255))
    img = np.transpose(img, (2, 0, 1)) # (H, W, 3) -> (3, H, W)
    img = np.expand_dims(img, axis=0) # (3, H, W) -> (1, 3, H, W)

    return img