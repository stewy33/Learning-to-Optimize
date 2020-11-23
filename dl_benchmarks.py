import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms


class Model(pl.LightningModule):
    def __init__(self, net, loss_fn, get_trajectory_weights=None):
        super().__init__()
        self.model = net
        self.loss_fn = loss_fn
        self.get_trajectory_weights = get_trajectory_weights

        self.optimizer = None
        self.trajectory = [self.get_trajectory_weights(self.model)]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean().item()

        if self.get_trajectory_weights is not None:
            self.trajectory.append(self.get_trajectory_weights(self.model))

        return {"loss": loss, "accuracy": accuracy}

    def training_epoch_end(self, outputs):
        self.logger.log_metrics(
            {
                "train_loss": np.mean([out["loss"] for out in outputs]),
                "train_accuracy": np.mean([out["accuracy"] for out in outputs]),
            },
            step=self.current_epoch,
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = self.loss_fn(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean().item()

        return {"loss": loss, "accuracy": accuracy}

    def validation_epoch_end(self, outputs):
        self.logger.log_metrics(
            {
                "val_loss": np.mean([out["loss"] for out in outputs]),
                "val_accuracy": np.mean([out["accuracy"] for out in outputs]),
            },
            step=self.current_epoch,
        )

    def configure_optimizers(self):
        return self.optimizer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def MNIST_MLP():
    # Define MNIST dataset, the required preprocessing, and its dataloaders
    tsfms = transforms.Compose([transforms.ToTensor()])

    train_set = datasets.MNIST(
        root="./data", train=True, download=True, transform=tsfms
    )
    test_set = datasets.MNIST(
        root="./data", train=False, download=True, transform=tsfms
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=512, shuffle=True, num_workers=5
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=5
    )

    # Make a 2-layer MLP with hidden layer size 300
    net = nn.Sequential(
        nn.Flatten(), nn.Linear(784, 300), nn.ReLU(), nn.Linear(300, 10)
    )
    # We will track the trajectories of two weights in the final layer for our plots
    def get_final_layer_weights(net):
        return net[3].weight[0, 0].item(), net[3].weight[0, 1].item()

    # Make our Pytorch Lightning model
    model = Model(
        net, loss_fn=F.cross_entropy, get_trajectory_weights=get_final_layer_weights
    )

    return model, train_loader, test_loader


"""
net = torchvision.models.squeezenet1_0()
    net.classifier._modules["1"] = nn.Conv2d(512, 10, kernel_size=(1, 1))
    net.num_classes = 10
    
    loss = F.cross_entropy
    
    return Model(net, loss)"""


def train(make_optimizer, model, train_loader, test_loader, hyperparams):
    optimizer = make_optimizer(model.parameters(), **hyperparams)
    model.set_optimizer(optimizer)

    trainer = pl.Trainer(logger=TensorBoardLogger("tb_logs", name="mlp"))
    trainer.fit(model, train_loader, test_loader)
