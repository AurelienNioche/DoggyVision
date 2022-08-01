import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tqdm import tqdm


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.Sigmoid(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.Sigmoid(),
            nn.Linear(hidden_layer_size, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y_hat = self.layers(x)
        # assert (y_hat >= 0.).any()
        return y_hat


class DoggyVisionDataset(Dataset):

    def __init__(self):

        self.X, self.y = self.load_data()
        self.y, self.y_kwargs_transform = self.normalize(self.y)

    def load_data(self):

        df = pd.read_csv("data/data.csv", index_col=0)
        df = df.drop(df[df.time == "/"].index)
        df = df.drop(df[df.video == "/"].index)
        df = df.dropna()
        df = df.reset_index()
        df.seconds = pd.to_numeric(df.seconds)

        df_labels = pd.read_csv("data/video_labels_05072022.csv")
        video_list = df.video.unique()
        video_labels = np.asarray(df_labels.video)

        for v in video_list:
            if v not in video_labels:
                raise ValueError(f"Error of labelling: '{v}' exists in the data, "
                                 f"but can't be found in the labels")

        label_categories = df_labels.columns.to_list()[1:]

        X = []
        y = []

        for index, row in df.iterrows():
            x = []
            for cat in label_categories:
                labels = sorted(df_labels[cat].unique())
                label = df_labels[df_labels["video"] == row.video][cat].values[0]

                for lb in labels:
                    x.append(lb == label)

                x.append(1.)  # Add unit cell

            X.append(x)
            y.append(row.seconds)

        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.float),

    @staticmethod
    def normalize(x):

        # x_mean = x.mean(dim=0, keepdim=True)
        # x_std = x.std(dim=0, unbiased=False, keepdim=True)
        # x -= x_mean
        # x /= x_std
        # print("x", x.mean(), x.std())
        x_min = x.min()
        x_max = x.max()
        x -= x_min
        x /= x_max
        return x, dict(x_min=x_min, x_max=x_max)

    def unnormalize(self, x, x_min, x_max):
        x *= x_max
        x += x_min
        # x*x_std + x_mean
        return x

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def main():

    torch.manual_seed(123)

    data = DoggyVisionDataset()

    n_obs = len(data)
    print("n observation", n_obs)

    n_training = int(0.80*n_obs)
    n_val = n_obs - n_training
    training_data, val_data = torch.utils.data.random_split(data, [n_training, n_val])

    # batch_size = len(training_data)

    model = NeuralNetwork(input_size=len(data.X[0]),
                          output_size=1,
                          hidden_layer_size=512)

    dataloader = DataLoader(training_data, batch_size=64)
    val_dataloader = DataLoader(val_data, batch_size=len(val_data))

    loss_fn = torch.nn.MSELoss()

    for batch, (X, y) in enumerate(val_dataloader):

        pred = model(X)[:, 0]

        unscaled_pred = data.unnormalize(pred, **data.y_kwargs_transform)
        unscaled_y = data.unnormalize(y, **data.y_kwargs_transform)

        print(f"Average abs error validation: {(unscaled_pred - unscaled_y).abs().mean():.2f}")

    optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())

    n_epochs = 100
    hist_loss = []

    for _ in tqdm(range(n_epochs)):
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)[:, 0]
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            hist_loss.append(loss.item())

    # for batch, (X, y) in enumerate(dataloader):
    #     pred = model(X)[:, 0]
    #     unscaled_pred = data.unnormalize(pred, **data.y_kwargs_transform)
    #     unscaled_y = data.unnormalize(y, **data.y_kwargs_transform)
    #     print(torch.abs(unscaled_y - unscaled_pred))

    for batch, (X, y) in enumerate(val_dataloader):
        pred = model(X)[:, 0]

        unscaled_pred = data.unnormalize(pred, **data.y_kwargs_transform)
        unscaled_y = data.unnormalize(y, **data.y_kwargs_transform)

        print(f"Average abs error validation: {(unscaled_pred - unscaled_y).abs().mean():.2f}")

    os.makedirs("fig", exist_ok=True)
    fig, ax = plt.subplots()
    ax.set_title(f"Loss")
    ax.plot(hist_loss)
    plt.savefig(f"fig/hist_loss.pdf")


if __name__ == '__main__':
    main()
