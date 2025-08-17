'''
    Frequency-aware NBRDF, IEEE MLSP 2025.
    
        Peter/Zheyuan Hu, Chenliang Zhou, 2024-2025.
''' 
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available()
                    #   else torch.device("mps") if
                    #   torch.backends.mps.is_available()
                      else "cpu")


def xyz2sph(x, y, z):
    r2_xy = x ** 2 + y ** 2
    r = np.sqrt(r2_xy + z ** 2)
    theta = np.arctan2(np.sqrt(r2_xy), z)
    phi = np.arctan2(y, x)
    return np.array([r, theta, phi])


def rvectors_to_rsph(hx, hy, hz, dx, dy, dz):
    half_sph = xyz2sph(hx, hy, hz)
    diff_sph = xyz2sph(dx, dy, dz)
    return half_sph, diff_sph


def brdf_to_rgb(rvectors, brdf):
    hx = torch.reshape(rvectors[:, :, 0], (-1, 1))
    hy = torch.reshape(rvectors[:, :, 1], (-1, 1))
    hz = torch.reshape(rvectors[:, :, 2], (-1, 1))
    dx = torch.reshape(rvectors[:, :, 3], (-1, 1))
    dy = torch.reshape(rvectors[:, :, 4], (-1, 1))
    dz = torch.reshape(rvectors[:, :, 5], (-1, 1))

    theta_h = torch.atan2(torch.sqrt(hx ** 2 + hy ** 2), hz)
    theta_d = torch.atan2(torch.sqrt(dx ** 2 + dy ** 2), dz)
    phi_d = torch.atan2(dy, dx)
    wiz = torch.cos(theta_d) * torch.cos(theta_h) - \
        torch.sin(theta_d) * torch.cos(phi_d) * torch.sin(theta_h)
    rgb = brdf * torch.clamp(wiz, 0, 1)
    return rgb


