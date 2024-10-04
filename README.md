# Outlier Detection through DDPM

SLURM setup at `job.sh` and `submit.sh`

Run `./submit.sh` to submit a SLURM job

Run `squeue --me` to see your job status

Run `tail -f out/#filename{.err/.out}` to seee the error and dutput logs

Arguments for model hyperparameters set at `job.sh`

Model checkpoints set at `model_checkpoints` ddireectory


## Available Arguments
Here's a list of all the arguments you can modify, along with their default values and descriptions:

`--mode`: Set to 'train' (default) or 'eval' to specify the mode of operation.

`--device`: Specify the device for training, default is 'cuda'.

`--epochs`: The number of training epochs, default is 20.

`--batch_size`: Size of each training batch, default is 64.

`--learning_rate`: The learning rate for the optimizer, default is 0.0004

`--ddpm_timesteps`: Number of timesteps for the DDPM process, default is 1000.

`--cosine_warmup_steps`: Warmup steps for the cosine learning rate scheduler, default is 500

`--cosine_total_training_steps`: Total training steps for the cosine learning rate scheduler, default is 10000.

### Examples

#### Train a model with custom settings:

```bash
#job.sh content
python diffuse/diiffuseMain.py --mode train --epochs 50 --batch_size 32 --learning_rate 0.001
```

#### Evaluate a model:

```bash
python diffuse/diiffuseMain.py --mode eval --epochs 50 --batch_size 32 --learning_rate 0.001

```

###### Additional Information: Ensure that the device specified is available and correctly configured in your environment (CUDA for NVIDIA GPUs). Adjust the learning rate and batch size based on your system's capabilities and the specific requirements of your model training.
