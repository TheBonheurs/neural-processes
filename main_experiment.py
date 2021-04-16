import json
import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from gru import GRUNet
from utils import context_target_split
from torch.utils.data import DataLoader
from datasets import mnist, celeba, SineData
from neural_process import NeuralProcessImg
from neural_process import NeuralProcess
from time import strftime
from training import NeuralProcessTrainer
from math import pi

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
#if config["dataset"] == "mnist":
#   data_loader, _ = mnist(batch_size=batch_size, size=img_size[1])
#elif config["dataset"] == "celeba":
#   data_loader = celeba(batch_size=batch_size, size=img_size[1])

#np_img = NeuralProcessImg(img_size, r_dim, z_dim, h_dim).to(device)

gru = GRUNet(50, 256, 50, 1)
hidden = gru.init_hidden(batch_size)
input_data = NeuralProcess(1, 1, 50, 50, 50, gru, hidden)

optimizer = torch.optim.Adam(input_data.parameters(), lr=config["lr"])

np_trainer = NeuralProcessTrainer(device, input_data, optimizer,
                                  num_context_range, num_extra_target_range,
                                  print_freq=100)

for epoch in range(epochs):
    print("Epoch {}".format(epoch + 1))
    np_trainer.train(data_loader, 1)

    # Save losses at every epoch
    with open(directory + '/losses.json', 'w') as f:
        json.dump(np_trainer.epoch_loss_history, f)

    # Save model at every epoch
    torch.save(np_trainer.neural_process.state_dict(), directory + '/model.pt')

    if epoch % 50 == 0:

        if epoch == 0:
            for batch in data_loader:
                break
            x, y = batch
            x_context, y_context, _, _ = context_target_split(x[0:1], y[0:1], 4, 4)

            x_target = torch.Tensor(np.linspace(-pi, pi, 100))
            x_target = x_target.unsqueeze(1).unsqueeze(0)

        input_data.training = False

        for i in range(64):
            # Neural process returns distribution over y_target
            p_y_pred = input_data(x_context, y_context, x_target)
            # Extract mean of distribution
            mu = p_y_pred.loc.detach()
            plt.plot(x_target.numpy()[0], mu.numpy()[0], alpha=0.05, c='b')

        input_data.training = True

        plt.scatter(x_context[0].numpy(), y_context[0].numpy(), c='k')
        plt.show()

