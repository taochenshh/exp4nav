import numpy as np
import torch


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


def explained_variance(ypred, y):
    """
    *** copied from openai/baselines ***
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary == 0 else 1 - np.var(y - ypred) / vary


def imagenet_rgb_preprocess(imgs, device=torch.device('cpu')):
    if not isinstance(imgs, torch.Tensor):
        imgs = torch.from_numpy(imgs).float()
    if len(imgs.shape) == 4:
        imgs = imgs.unsqueeze(dim=0)
    imgs = imgs.permute(0, 1, 4, 2, 3).to(device)  # (Seq, N, C, H, W)
    imgs = imgs / 255.0
    rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
    rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device)
    rgb_mean = rgb_mean[None, None, :, None, None]
    rgb_std = rgb_std[None, None, :, None, None]
    imgs = (imgs - rgb_mean) / rgb_std
    return imgs


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """ copied from openai/baselines
        Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class ExpDecaySchedule(object):
    def __init__(self, lam1=0.1, lam2=0.95):
        """ copied from openai/baselines
        decay in lam1*lam2^t
        """
        self.lam1 = lam1
        self.lam2 = lam2

    def value(self, t):
        return self.lam1 * self.lam2 ** t
