import numpy as np
from numpy.fft import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import sys
from tqdm import tqdm

def to_freq(x):
    x =  x.cpu().detach().numpy().squeeze()
    return np.fft.fftshift(mag(x))


def mag(x):
    '''Magnitude
    x[..., 0] is real part
    x[..., 1] is imag part
    '''
    return np.sqrt(np.square(x[..., 0]) + np.square(x[..., 1]))


def freq_band(n=28, band_center=0.3, band_width=0.05):
    freq_arr = fftshift(fftfreq(n))
    freq_arr /= np.max(np.abs(freq_arr))
    mask_bandpass = np.zeros((n,n))

    for r in range(n):
        for c in range(n):
            dist = np.sqrt(freq_arr[r]**2 + freq_arr[c]**2)
            if dist > band_center - band_width / 2 and dist < band_center + band_width / 2:
                mask_bandpass[r, c] = 1
    
    return mask_bandpass