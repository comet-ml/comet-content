from comet_ml import Experiment

import torch
import torch.nn.functional as F
import torch.distributed as dist

from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP

from collections import OrderedDict
from tqdm import tqdm

EPOCHS = 5
BS = 64

experiment = Experiment(workspace="cometpublic", project_name="histograms")
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = datasets.MNIST("/tmp", download=True, train=True, transform=transform)
valset = datasets.MNIST("/tmp", download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BS, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=BS, shuffle=True)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(
    OrderedDict(
        [
            ("linear0", nn.Linear(input_size, hidden_sizes[0])),
            ("activ0", nn.ReLU()),
            ("linear1", nn.Linear(hidden_sizes[0], hidden_sizes[1])),
            ("activ1", nn.ReLU()),
            ("linear2", nn.Linear(hidden_sizes[1], output_size)),
            ("output", nn.LogSoftmax(dim=1)),
        ]
    )
)
optimizer = optim.Adam(model.parameters())


def update_gradient_map(gradmap, model):
    for name, layer in zip(model._modules, model.children()):
        if "activ" in name:
            continue

        if not hasattr(layer, "weight"):
            continue

        wname = "%s/%s.%s" % ("gradient", name, "weight")
        bname = "%s/%s.%s" % ("gradient", name, "bias")

        gradmap.setdefault(wname, 0)
        gradmap.setdefault(bname, 0)

        gradmap[wname] += layer.weight.grad
        gradmap[bname] += layer.bias.grad

    return gradmap


def log_histogram_3d(gradmap, step):
    for k, v in gradmap.items():
        experiment.log_histogram_3d(v, name=k, step=step)


def log_weights(model, step):
    for name, layer in zip(model._modules, model.children()):
        if "activ" in name:
            continue

        if not hasattr(layer, "weight"):
            continue

        wname = "%s.%s" % (name, "weight")
        bname = "%s.%s" % (name, "bias")

        experiment.log_histogram_3d(layer.weight, name=wname, step=step)
        experiment.log_histogram_3d(layer.bias, name=bname, step=step)


def train(model, dataloader, epoch):
    gradmap = {}
    steps_per_epoch = len(dataloader.dataset) // BS

    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(dataloader)):
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        pred = model(data)

        loss = F.nll_loss(pred, target)
        loss.backward()

        gradmap = update_gradient_map(gradmap, model)
        optimizer.step()

    # scale gradients
    for k, v in gradmap.items():
        gradmap[k] = v / steps_per_epoch

    log_histogram_3d(gradmap, epoch * steps_per_epoch)
    log_weights(model, epoch * steps_per_epoch)


def main():
    for epoch in range(1, EPOCHS + 1):
        train(model, trainloader, epoch)


if __name__ == "__main__":
    main()
