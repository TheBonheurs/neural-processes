import json
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from gru import GRUNet
from torch.utils.data import DataLoader
from datasets import mnist, celeba, SineData
from neural_process import NeuralProcessImg
from neural_process import NeuralProcess
from time import strftime
from training import NeuralProcessTrainer
from numpyencoder import NumpyEncoder
from math import pi


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main_experiment.py <path_to_config>"))
config_path = sys.argv[1]

# Create a folder to store experiment results
timestamp = strftime("%Y-%m-%d_%H-%M")
directory = "results_{}".format(timestamp)
if not os.path.exists(directory):
    os.makedirs(directory)

# Open config file
with open(config_path) as config_file:
    config = json.load(config_file)

# Save config file in experiment directory
with open(directory + '/config.json', 'w') as config_file:
    json.dump(config, config_file)

img_size = config["img_size"]
batch_size = config["batch_size"]
r_dim = config["r_dim"]
h_dim = config["h_dim"]
z_dim = config["z_dim"]
num_context_range = config["num_context_range"]
num_extra_target_range = config["num_extra_target_range"]
epochs = config["epochs"]

dataset = SineData(amplitude_range=(-1., 1.), shift_range=(-.5, .5), num_points=400, num_samples=800)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
if config["dataset"] == "mnist":
   data_loader, _ = mnist(batch_size=batch_size, size=img_size[1])
elif config["dataset"] == "celeba":
   data_loader = celeba(batch_size=batch_size, size=img_size[1])

gru = GRUNet(r_dim, h_dim, r_dim, 1)
hidden = gru.init_hidden(batch_size)
input_data = NeuralProcess(1, 1, 50, 50, 50, gru, hidden)

np_img = NeuralProcessImg(img_size, r_dim, z_dim, h_dim, gru, hidden).to(device)

optimizer = torch.optim.Adam(input_data.parameters(), lr=config["lr"])

np_trainer = NeuralProcessTrainer(device, np_img, optimizer,
                                  num_context_range, num_extra_target_range,
                                  print_freq=100)
model = None

for epoch in range(epochs):
    print("Epoch {}".format(epoch + 1))
    np_trainer.train(data_loader, 1)
    # Save losses at every epoch
    with open(directory + '/losses.json', 'w') as f:
        json.dump(np_trainer.epoch_loss_history, f)
    # Save mu's and sigmas at every epoch
    # with open(directory + '/mu.json', 'w') as f:
    #     json.dump(np_trainer.mu_list, f, cls=NumpyEncoder)
    # with open(directory + '/sigma.json', 'w') as f:
    #     json.dump(np_trainer.sigma_list, f, cls=NumpyEncoder)
        
    # Save model at every epoch
    torch.save(np_trainer.neural_process.state_dict(), directory + '/model.pt')
    model = np_trainer.neural_process

import imageio
from torchvision.utils import make_grid

# Read images into torch.Tensor
all_imgs = torch.zeros(8, 3, 32, 32)

dat = iter(data_loader)

for i in range(8):
    img = imageio.imread('imgs/celeba-examples/{}.png'.format(i + 1))
    all_imgs[i] = torch.Tensor(img.transpose(2, 0, 1) / 255.)

# Visualize sample on a grid
img_grid = make_grid(all_imgs, nrow=4, pad_value=1.)
plt.imshow(img_grid.permute(1, 2, 0).numpy())
plt.show()

# Select one of the images to perform inpainting
img = all_imgs[0]

# Define a binary mask to occlude image. For Neural Processes,
# the context points will be defined as the visible pixels
context_mask = torch.zeros((32, 32)).byte()
context_mask[:16, :] = 1  # Top half of pixels are visible

# Show occluded image
occluded_img = img * context_mask.float()
plt.imshow(occluded_img.permute(1, 2, 0).numpy())
plt.show()

from utils import inpaint

num_inpaintings = 8  # Number of inpaintings to sample from model
all_inpaintings = torch.zeros(num_inpaintings, 3, 32, 32)

# Load config file for celebA model
folder = 'trained_models/celeba'
config_file = folder + '/config.json'
model_file ='/results_2021-04-16_18-40/model.pt'

model.load_state_dict(torch.load(folder + model_file, map_location=lambda storage, loc: storage))

# Sample several inpaintings
for i in range(num_inpaintings):
    all_inpaintings[i] = inpaint(model, img, context_mask, device)

# Visualize inpainting results on a grid
inpainting_grid = make_grid(all_inpaintings, nrow=4, pad_value=1.)
print("HALLOOOOOOOOOOO")
plt.imshow(inpainting_grid.permute(1, 2, 0).numpy())
plt.show()