from collections import defaultdict

import torch
import torch.nn.functional as F

from diffuseNew.utils.sampling import denoise_image, q_sample

import logging

logging.basicConfig(level=logging.INFO,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
                    handlers=[
                        logging.FileHandler("training.log"),  # Log to a file
                        logging.StreamHandler()  # Also log to the console
                    ])

logger = logging.getLogger(__name__)

def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def compute_reconstruction_error(original, reconstructed):
    """
    Compute Mean Squared Error between the original and reconstructed images.
    """
    return F.mse_loss(original, reconstructed, reduction='none').mean(dim=[1, 2, 3])


def reconstruction_error_by_class(model, test_dataloader, device):
    class_reconstruction_errors = defaultdict(list)
    for batch in test_dataloader:
        images = batch["pixel_values"].to(device)  # Get images from the batch
        labels = batch["label"].to(device)  # Get the class labels for the batch
        batch_size = images.size(0)

        # Choose a timestep (e.g., 150 for halfway through the process)
        timestep = torch.tensor([150] * batch_size).to(device)

        # Add noise to the images at the given timestep
        noisy_images = q_sample(images, timestep)

        # Denoise the images using the diffusion model
        reconstructed_images = denoise_image(model, noisy_images, timesteps=300)

        # Calculate reconstruction error for each image in the batch
        errors = compute_reconstruction_error(images, reconstructed_images)

        # Accumulate errors for each class
        for i in range(batch_size):
            label = labels[i].item()
            class_reconstruction_errors[label].append(errors[i].item())

        # Calculate the average reconstruction error for each class and print it neatly
        avg_class_reconstruction_errors = {label: sum(errors) / len(errors) for label, errors in
                                           class_reconstruction_errors.items()}

        # Print the results neatly
        logger.info("Average Reconstruction Error for Each Class each loop:")
        for label, avg_error in avg_class_reconstruction_errors.items():
            logger.info(f"Class {label}: {avg_error:.4f}")

    # Calculate average reconstruction error for each class
    avg_class_reconstruction_errors = {label: torch.tensor(errors).mean().item()
                                       for label, errors in class_reconstruction_errors.items()}

    # Print the average reconstruction error for each class
    for label, avg_error in avg_class_reconstruction_errors.items():
        logger.info(f"Average Reconstruction Error for Class {label}: {avg_error:.4f}")