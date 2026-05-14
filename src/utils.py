import torch

def fast_mmd(real, fake, sigmas=[0.05, 0.1, 0.7, 5.0, 10.0]):
    # Vectorized distance calculation: ||x-y||^2 = ||x||^2 + ||y||^2 - 2x*y.T
    x_sq = torch.sum(real**2, dim=1).unsqueeze(1)
    y_sq = torch.sum(fake**2, dim=1).unsqueeze(0)

    dist_xx = x_sq + x_sq.T - 2 * torch.mm(real, real.T)
    dist_yy = y_sq + y_sq.T - 2 * torch.mm(fake, fake.T)
    dist_xy = x_sq + y_sq - 2 * torch.mm(real, fake.T)
    total_mmd=0

    for s in sigmas:
        gamma = 1.0 / (2 * s**2)
        k_xx = torch.exp(-dist_xx * gamma)
        k_yy = torch.exp(-dist_yy * gamma)
        k_xy = torch.exp(-dist_xy * gamma)

        total_mmd += (k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())

    return total_mmd

def boundary_loss(fake_data, real_min, real_max):
    low_penalty = torch.relu(real_min - fake_data)
    high_penalty = torch.relu(fake_data - real_max)
    return torch.mean(low_penalty + high_penalty)

