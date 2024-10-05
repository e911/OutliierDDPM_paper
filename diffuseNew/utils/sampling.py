import torch
from tqdm import tqdm

from diffuseNew.utils.lib import *


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


@torch.no_grad()
def denoise_image(model, noisy_image, timesteps=300):
    """
    Reconstructs a denoised image from a noisy image using the diffusion model.
    model: the trained diffusion model
    noisy_image: noisy input image
    timesteps: number of timesteps for denoising
    """
    device = next(model.parameters()).device  # Ensure model and images are on the same device
    noisy_image = noisy_image.to(device)
    img = noisy_image.clone()

    for i in reversed(range(0, timesteps)):
        # Create a tensor of the current timestep for the batch
        t = torch.full((noisy_image.size(0),), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, t_index=i)  # Call your sampling function from sampling.py

    return img