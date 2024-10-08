# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tVynUFJtMuz0IkznGrFLzkUM37_7XTNP
"""
import argparse
import os
# Commented out IPython magic to ensure Python compatibility.

from pathlib import Path

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image


from diffuseNew.models.loss import p_losses, reconstruction_error_by_class
from diffuseNew.models.unet import *
from diffuseNew.utils.sampling import sample, get_noisy_image, q_sample
from diffuseNew.utils.schedule import *
from diffuseNew.utils.transforms import transform, transform_dataset, transforms_dataset
from diffuseNew.utils.visualization import plot, plot_denoise_steps
from datasets import load_dataset, concatenate_datasets

torch.manual_seed(0)


import logging

logging.basicConfig(level=logging.INFO,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
                    handlers=[
                        logging.FileHandler("training.log"),  # Log to a file
                        logging.StreamHandler()  # Also log to the console
                    ])

logger = logging.getLogger(__name__)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def create_imbalanced_dataset(dataset, class_samples):
    subsets = []
    for label, max_samples in class_samples.items():
        # Filter the dataset for the specific label
        filtered_data = dataset.filter(lambda example: example['label'] == label)
        # Select a subset if max_samples is specified and less than the available examples
        if max_samples is not None and len(filtered_data) > max_samples:
            filtered_data = filtered_data.shuffle(seed=42).select(range(max_samples))
        subsets.append(filtered_data)
    # Concatenate all subsets into one dataset using the correct concatenate method
    return concatenate_datasets(subsets)

def load_train_data(batch_size):
    train_dataset = load_dataset("mnist", split='train')
    class_samples = {0: 6000, 1: 6000, 2: 6000, 3: 100, 4: 6000, 5: 6000, 6: 6000, 7: 6000, 8: 20, 9: 6000}
    train_dataset = create_imbalanced_dataset(train_dataset, class_samples)
    class_counts = {label: sum(1 for x in train_dataset['label'] if x == label) for label in range(10)}
    logger.info("Class counts in the imbalanced dataset:")
    for label, count in class_counts.items():
        logger.info(f"Class {label}: {count}")

    transformed_dataset = train_dataset.with_transform(transforms_dataset)
    train_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)

    return train_loader

def load_test_data(batch_size):
    test_dataset = load_dataset("mnist", split='test')
    # class_samples = {0: 6000, 1: 6000, 2: 50, 3: 100, 4: 5500, 5: 5500, 6: 5500, 7: 5500, 8: 6000, 9: 6000}
    # test_dataset = create_imbalanced_dataset(test_dataset, class_samples)
    # class_counts = {label: sum(1 for x in test_dataset['label'] if x == label) for label in range(10)}
    # logger.info("Class counts in the imbalanced dataset:")
    # for label, count in class_counts.items():
    #     logger.info(f"Class {label}: {count}")

    transformed_dataset = test_dataset.with_transform(transforms_dataset)

    test_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True)
    return test_loader

def train(config):

    # Setup parameters
    image_size = 28
    channels = config['channels']
    batch_size = config['batch_size']
    timesteps = config['timesteps']  # Number of diffusion timesteps
    epochs = config['epochs']  # Number of training epochs

    # Create a DataLoader for training data
    dataloader = load_train_data(batch_size)

    # Check a sample batch to ensure data format is correct
 # Should show 'pixel_values' and possibly 'labels'

    # Create a directory to save results and checkpoints
    results_folder = Path("./results")
    results_folder.mkdir(exist_ok=True)
    checkpoint_folder = Path("./checkpoints")
    checkpoint_folder.mkdir(exist_ok=True)
    save_and_sample_every = 1000

    # Set device to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4)
    )
    model.to(device)

    # Set up the optimizer
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])

    # Training loop
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0  # To accumulate epoch loss

        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()  # Reset gradients

            # Get batch data and move to device
            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Sample random timesteps
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # Calculate loss
            loss = p_losses(model, batch, t, loss_type="l2")

            # Print loss at every 100 steps
            if step % 100 == 0:
                logger.info(f"Step {step}: Loss = {loss.item():.4f}")

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Accumulate loss for the epoch
            epoch_loss += loss.item()

            # Save generated images periodically
            # if step != 0 and step % save_and_sample_every == 0:
            #     milestone = step // save_and_sample_every
            #     batches = num_to_groups(4, batch_size)
            #     all_images_list = [sample(model, batch_size=n, channels=channels) for n in batches]
            #     all_images = torch.cat(all_images_list, dim=0)
            #     all_images = (all_images + 1) * 0.5  # Scale back to [0, 1]
            #     save_image(all_images, str(results_folder / f'sample-{epoch}-{milestone}.png'), nrow=6)

        # Calculate and print average epoch loss
        avg_epoch_loss = epoch_loss / len(dataloader)
        logger.info(f"Average Loss for Epoch {epoch + 1}: {avg_epoch_loss:.4f}")

        # Save model checkpoint after each epoch
        checkpoint_path = checkpoint_folder / f"model_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, checkpoint_path)
        logger.info(f"Checkpoint saved at {checkpoint_path}")

def load_model(checkpoint_path, image_size=28, channels=1):
    """
    Load a saved model checkpoint for evaluation.

    Args:
        checkpoint_path (str or Path): Path to the checkpoint file.
        image_size (int): Size of the image input (default is 28 for MNIST).
        channels (int): Number of input channels (1 for grayscale images).
        device (str): Device to load the model on ('cuda' or 'cpu').

    Returns:
        model (torch.nn.Module): The loaded model with state_dict applied.
    """
    # Define the model architecture
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2, 4)
    )
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load the state dictionaries
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded model from checkpoint: {checkpoint_path}")

    return model

def eval_recon_loss(config):
    image_size = 28
    channels = config['channels']
    batch_size = config['batch_size']
    epoch = config['epochs']
    checkpoint_folder = Path("./checkpoints")
    checkpoint_path = checkpoint_folder / f"model_epoch_{epoch + 1}.pth"
    test_dataloader = load_test_data(batch_size=batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(checkpoint_path, image_size, channels)
    reconstruction_error_by_class(model, test_dataloader, device=device)


# def plot_noisy_image(timestep, image):
#     noisy_dir = f"./noisy_images"
#     if not os.path.exists(noisy_dir):
#         os.makedirs(noisy_dir)
#         logger.info(f"Created checkpoint directory at {noisy_dir}")
#
#     # url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
#     # image = Image.open(requests.get(url, stream=True).raw)  # PIL image of shape HWC
#     x_start = transform(image).unsqueeze(0)
#     print(x_start.shape)
#     t = torch.tensor([timestep])
#     a = get_noisy_image(x_start, t)
#     print(a)
#     plot([a,], image, save_path=f'{noisy_dir}/noisy_image.png')
#
#
# def plot_noisy_image_timesteps(image, step=15, timesteps=300):
#     checkpoint_dir = f"./noisy_images"
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#         logger.info(f"Created checkpoint directory at {checkpoint_dir}")
#
#     x_start = transform(image).unsqueeze(0)
#     for timestep in range(0, timesteps + 1, step):
#         plot([get_noisy_image(x_start, torch.tensor([timestep]))], x_start, save_path=f'{checkpoint_dir}/noisy_image_{timestep}.png')

def per_images(n):
  image = []
  test_dataloader = load_dataset("mnist", split='test')
  for i in range(10):
    count = 0
    for each in test_dataloader:
      if each['label'] == i:
        image.append(each)
        count = count + 1
      if count == n:
          break
  return image

def noise_denoise_steps(config):
    training_dir = f"./denosing_steps"
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
        logger.info(f"Created sample directory at {training_dir}")
    image_size = 28
    channels = config['channels']
    timesteps = config['timesteps']  # Number of diffusion timesteps
    epoch = config['epochs']
    steps = config['steps']
    images = per_images(config['n'])
    checkpoint_folder = Path("./checkpoints")
    checkpoint_path = checkpoint_folder / f"model_epoch_{epoch}.pth"
    model = load_model(checkpoint_path, image_size, channels)
    for each in images:
        img = transform(each['image']).unsqueeze(0)
        plot([q_sample(img, torch.tensor([t])) for t in [0, 50, 150, 200, timesteps]], image_size, channels, img, True, save_path=f"{training_dir}/noise_{each['label']}.png")
        plot_denoise_steps(model, each, timesteps=timesteps, image_size=image_size, channels=channels, steps=steps)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Configuration for Diffusion Model')
    parser.add_argument('--mode', type=str, default='train', help='Train or Eval')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--channels', type=int, default=1, help='Number of channels 1/3')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0004, help='Optimizer learning rate')
    parser.add_argument('--ddpm_timesteps', type=int, default=400, help='Number of DDPM timesteps')
    # parser.add_argument('--cosine_warmup_steps', type=int, default=500, help='Warmup steps for learning rate scheduler')
    # parser.add_argument('--cosine_total_training_steps', type=int, default=10000, help='Total training steps for learning rate scheduler')
    parser.add_argument('--n', type=int, default=1, help='Number of images per class')
    parser.add_argument('--steps', type=int, default=50, help='Skip steps in saving image')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    logger.info("Running with the following configuration:")
    logger.info(f"Mode: {args.mode}")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    config = {
        'device': args.device,
        'epochs': args.epochs,
        'channels': args.channels,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
    #     'class_counts': {0: 5000, 1:10, 2: 4400, 3: 4000, 4: 4800, 5: 4400, 6: 4200, 7: 4000, 8: 4000, 9: 10},
        'timesteps': args.ddpm_timesteps,
    #     'cosine_warmup_steps': args.cosine_warmup_steps,
    #     'cosine_warmup_training_steps': args.cosine_total_training_steps,
    #     'learning_rate': args.learning_rate,
        'n': args.n,
        'steps': args.steps
    }

    if args.mode == 'train':
        train(config)
    # elif args.mode == 'eval':
    #     eval_recon(config)
    elif args.mode == 'eval_diff':
        noise_denoise_steps(config)
    elif args.mode == 'eval_classwise':
        eval_recon_loss(config)
    # elif args.mode == 'one':
    #     eval_one(config)
    else:
        logger.error("Invalid mode")


