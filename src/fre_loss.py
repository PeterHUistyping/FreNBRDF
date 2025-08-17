'''
    Frequency-aware NBRDF, IEEE MLSP 2025.

        Peter/Zheyuan Hu, Chenliang Zhou, 2024-2025.

    Improved frequency extraction method with kNN for material data reconstruction.

    Reference: 
    - HyperBRDF & FrePolad ECCV 2024.
''' 
import torch_harmonics as th
import torch, numpy as np
from cv2 import getGaussianKernel
from utils import brdf_to_rgb, get_device, rvectors_to_rsph, brdf_to_rgb


device = get_device()


def extract_high_freq(batch, nlat, nlon, lmax=49, sigma=50):
    '''
        Extracts frequency-related features from a batch of data.
    
    input:
        batch: torch.Tensor, material of shape (batch_size, nlat * nlon), 
            where `nlat` and `nlon` are related to the spherical grid.
        
        lmax: int, the maximum degree of spherical harmonics to consider.
        
        sigma: float, the standard deviation for the Gaussian kernel.

    output:
        torch.Tensor: Concatenated coefficients after extraction.

    adapted from:
        FrePolad, ECCV 2024.
    '''    

    # data = extract_to_sphere(batch, nlat, nlon, device=device)
    data = batch.reshape(batch.shape[0], nlon, nlat)
    sht = th.RealSHT(nlon, nlat, lmax=lmax + 1, grid="equiangular").to(device)
    coeffs = sht(data)[:, :, :lmax + 1]
    coeffs = torch.stack([coeffs.real, coeffs.imag], dim=1)
    weights = torch.from_numpy(getGaussianKernel(coeffs.shape[2] * 2 - 1, sigma)[coeffs.shape[2] - 1::-1].reshape(-1).copy()).to(device, torch.float32)
    if coeffs.shape[3] == 1:
       last_dim = True
       coeffs = coeffs[:, :, :, 0]
    else:
       last_dim = False
    coeffs *= weights / weights[-1]

    all_coeffs = []
    for i in range(lmax + 1):
        if not last_dim:
          all_coeffs.append(coeffs[:, 0, i, :i + 1])
          all_coeffs.append(coeffs[:, 1, i, :i])
        else:
          all_coeffs.append(coeffs[:, 0, i])
          all_coeffs.append(coeffs[:, 1, i])
    if not last_dim:
      freq_info = torch.cat(all_coeffs, dim=1)
    else:
      freq_info = torch.cat(all_coeffs, dim=0)
    return freq_info


def chordal_distance(theta, phi, grid_lat, grid_lon):
    '''
        Computes the chordal distance between two spherical coordinates.
    '''
    pre_dist = torch.cos(grid_lat) * torch.cos(phi) * torch.cos(grid_lon - theta) + torch.sin(grid_lat) * torch.sin(phi)
    return 2 - 2 * pre_dist


def extract_to_sphere(theta, phi, nlat, nlon, device, knn_k = 2, knn_sigma = 0.05):
    ''' 
        Extracts the spherical grid from the input data.
        input:
          nlat: int, the number of latitudes. (mapped to phi)
          nlon: int, the number of longitudes. (mapped to theta)

        chordal distance

    '''
    ngrid = nlat * nlon
    phi = torch.from_numpy(phi).to(device).reshape(1, -1)
    theta = torch.from_numpy(theta).to(device).reshape(1, -1)

    grid_lat = torch.linspace(0, np.pi, nlat, device=device).to(device)
    grid_lon = torch.linspace(0, np.pi / 2, nlon, device=device).to(device)

    # row vector
    grid_lon = torch.broadcast_to(grid_lon, (nlat, nlon))
    # col vector
    grid_lat = torch.broadcast_to(grid_lat.view(nlat, 1), (nlat, nlon))

    grid_lon = grid_lon.reshape(ngrid, 1)
    grid_lat = grid_lat.reshape(ngrid, 1)
    # compute pairwise distances between grid_lon, grid_lat and theta, phi
    squared_dist = chordal_distance(theta, phi, grid_lat, grid_lon)

    _, inds = squared_dist.topk(k=knn_k, dim=1, largest=False)  # inds: [B, knn_k, ngrid]
    gaussian_coeffs = (-squared_dist.gather(dim=1, index=inds) / (2 * knn_sigma ** 2)).exp()

    # zero_threshold = 1e-10
    # zero_inds = (gaussian_coeffs < zero_threshold)
    # gaussian_coeffs = gaussian_coeffs - zero_inds * gaussian_coeffs
    # zero_rows_inds = zero_inds.all(dim=1, keepdim=True).expand(-1, knn_k, -1)
    # gaussian_coeffs = gaussian_coeffs + zero_rows_inds * (1 / knn_k)
    gaussian_coeffs = gaussian_coeffs / gaussian_coeffs.sum(dim=1, keepdim=True)

    return inds, gaussian_coeffs


def fre_updated_batch(batch, inds, gaussian_coeffs, ngrid):
    # fetch (ngrid, knn_k) elements from batch in vectorized 
    new_batch = batch[0][inds.reshape(-1)].reshape(inds.shape[0], inds.shape[1])
    return (new_batch * gaussian_coeffs).sum(dim=1).reshape(1, ngrid)


def fre_loss(batch1, batch2, coords, lmax=49, sigma=50):

    # nlat = 2 * lmax + 2
    nlat = lmax * 2
    nlon = lmax
    # nlat += 1
    # nlon += 1

    freq_loss_total = 0
    assert batch1.shape == batch2.shape 

    coords = coords[0].detach().cpu().numpy()
    
    half_sph, diff_sph = rvectors_to_rsph(coords[...,0], coords[...,1], coords[...,2], coords[...,3], coords[...,4], coords[...,5])

    # note that half_phi doesn't matter, 0 
    half_r, half_theta, half_phi = half_sph
    diff_r, diff_theta, diff_phi = diff_sph
    # clip theta to [0, pi/2] by remainder
    half_theta = np.remainder(half_theta, np.pi / 2)
    diff_theta = np.remainder(diff_theta, np.pi / 2)
    # clip phi to [0, pi]
    half_phi = np.remainder(half_phi, np.pi)
    diff_phi = np.remainder(diff_phi, np.pi)

    for i in range(0, 1):
        if i == 0:
            inds, gaussian_coeffs = extract_to_sphere(diff_theta, diff_phi, nlat, nlon, device)
        else:
            nlat = 1
            inds, gaussian_coeffs = extract_to_sphere(half_theta, half_phi, 1, nlon, device)

        batch1r = fre_updated_batch(batch1[..., 0], inds, gaussian_coeffs, nlat * nlon)
        batch1g = fre_updated_batch(batch1[..., 1], inds, gaussian_coeffs, nlat * nlon)
        batch1b = fre_updated_batch(batch1[..., 2], inds, gaussian_coeffs, nlat * nlon)
        batch2r = fre_updated_batch(batch2[..., 0], inds, gaussian_coeffs, nlat * nlon)
        batch2g = fre_updated_batch(batch2[..., 1], inds, gaussian_coeffs, nlat * nlon)
        batch2b = fre_updated_batch(batch2[..., 2], inds, gaussian_coeffs, nlat * nlon)

        coeffs1 = extract_high_freq(batch1r, nlat, nlon, lmax=lmax, sigma=sigma)
        coeffs2 = extract_high_freq(batch2r, nlat, nlon, lmax=lmax, sigma=sigma)
        freq_loss_total += torch.nn.functional.mse_loss(coeffs1, coeffs2)

        coeffs1 = extract_high_freq(batch1g, nlat, nlon, lmax=lmax, sigma=sigma)
        coeffs2 = extract_high_freq(batch2g, nlat, nlon, lmax=lmax, sigma=sigma)

        freq_loss_total += torch.nn.functional.mse_loss(coeffs1, coeffs2)

        coeffs1 = extract_high_freq(batch1b, nlat, nlon, lmax=lmax, sigma=sigma)
        coeffs2 = extract_high_freq(batch2b, nlat, nlon, lmax=lmax, sigma=sigma)
        freq_loss_total += torch.nn.functional.mse_loss(coeffs1, coeffs2)

    return freq_loss_total


# main function:
if __name__ == "__main__":


    def fre_hypernetwork_loss(kl, fw, model_output, gt, eta=0.1):
        '''
            Please add the following components to HyperBRDF, ECCV 2024.
        '''
        freq_loss = eta * fre_loss(
                brdf_to_rgb(model_output['model_in'], model_output['model_out']), brdf_to_rgb(model_output['model_in'], gt['amps']), model_output['model_in'])
