import os

import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
import requests

from diffuseNew.main import logger
from diffuseNew.utils.sampling import sample
from diffuseNew.utils.transforms import transform


def plot(imgs, original_image=None, with_orig=False, row_title=None, save_path=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(figsize=(200,200), nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [original_image] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)


def visualize_reconstruction(original_image, noisy_image, reconstructed_image, idx, save_path=None):
    """
    Display original, noisy, and reconstructed images side by side for visual comparison.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Display original image
    ax[0].imshow(original_image.squeeze().cpu().numpy(), cmap="gray")
    ax[0].set_title(f"Original Image {idx}")
    ax[0].axis('off')

    # Display noisy image
    ax[1].imshow(noisy_image.squeeze().cpu().numpy(), cmap="gray")
    ax[1].set_title(f"Noisy Image {idx}")
    ax[1].axis('off')

    # Display reconstructed image
    reconstructed_image = reconstructed_image.clamp(-1, 1)  # Clamp values for visualization
    reconstructed_image = (reconstructed_image + 1) * 0.5  # Scale to [0, 1]
    ax[2].imshow(reconstructed_image.squeeze().cpu().numpy(), cmap="gray")
    ax[2].set_title(f"Reconstructed Image {idx}")
    ax[2].axis('off')
    if save_path:
        plt.savefig(save_path)



def generate_sample(model, image_size, batch_size, channels, timesteps):
    training_dir = f"./training_steps"
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
        logger.info(f"Created training directory at {training_dir}")

    samples = sample(model, image_size=image_size, batch_size=batch_size, channels=channels)

    random_index = 5
    fig, axes = plt.subplots(1, int(timesteps/10), figsize=(int(timesteps/10) * 2, 2))

    # Remove spaces between the subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # Loop through each timestep and plot the corresponding image
    for i in range(timesteps):
        if i % 20 == 0:
          step = int(i/10)
          axes[step].imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray")
          axes[step].axis('off')  # Remove axis for a cleaner look

    # Display the entire row of images
    plt.tight_layout(pad=0)
    plt.savefig(f'{training_dir}/train.png')# Ensure no padding around the images
    plt.show()
