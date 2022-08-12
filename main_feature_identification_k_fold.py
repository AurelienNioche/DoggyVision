import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold

from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = nn.Linear(input_size, output_size)
        self.reset()

    def forward(self, x):
        return self.model(x)

    def reset(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()


class DoggyVisionDataset(Dataset):

    def __init__(self, cat_index):

        self.cat_index = cat_index

        self.X, self.y = self.load_data()
        self.y, self.y_kwargs_transform = self.normalize(self.y)

    def load_data(self):

        df = pd.read_csv("data/data.csv", index_col=0)
        df = df.drop(df[df.time == "/"].index)
        df = df.drop(df[df.video == "/"].index)
        df = df.dropna()
        df = df.reset_index()
        df.seconds = pd.to_numeric(df.seconds)

        df = df[df.seconds <= 50]

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
            for idx, cat in enumerate(label_categories):

                if idx != self.cat_index:
                    continue
                labels = sorted(df_labels[cat].unique())
                label = df_labels[df_labels["video"] == row.video][cat].values[0]

                for lb in labels:
                    x.append(lb == label)

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

    fig_folder = "fig/main_feature_identification_f_fold"
    os.makedirs(fig_folder, exist_ok=True)

    df_labels = pd.read_csv("data/video_labels_05072022.csv")
    label_categories = df_labels.columns.to_list()[1:]

    torch.manual_seed(123)

    traces = []

    for cat_index, cat in enumerate(label_categories):
        print(cat)

        dataset = DoggyVisionDataset(cat_index=cat_index)

        input_size = len(dataset.X[0])
        model = Model(input_size=input_size, output_size=1)

        loss_fn = torch.nn.MSELoss()

        n_obs = len(dataset)
        print("n observation", n_obs)

        k_folds = 20

        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=k_folds, shuffle=True)

        for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

            print('--------------------------------')
            print(f'FOLD {fold}')
            print('--------------------------------')

            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

            # Define data loaders for training and testing data in this fold
            trainloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=len(train_ids), sampler=train_subsampler)
            testloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=len(test_ids), sampler=test_subsampler)

            model.reset()

            # dataloader = DataLoader(training_data, batch_size=len(train_ids))
            # val_dataloader = DataLoader(val_data, batch_size=len(train_ids))

            for batch, (X, y) in enumerate(testloader):
                print("BEFORE LEARNING")

                pred = model(X)[:, 0]

                unscaled_pred = dataset.unnormalize(pred, **dataset.y_kwargs_transform)
                unscaled_y = dataset.unnormalize(y, **dataset.y_kwargs_transform)

                err_pred = (unscaled_pred - unscaled_y).abs().mean().item()

                traces.append(dict(cat=cat, err=err_pred, learning="before"))

                print(f"Average abs error validation: {err_pred:.2f}")

            optimizer = torch.optim.Adam(lr=0.01, params=model.parameters())

            n_epochs = 100
            hist_loss = []

            for _ in tqdm(range(n_epochs)):
                for batch, (X, y) in enumerate(trainloader):
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

            for batch, (X, y) in enumerate(testloader):
                print("AFTER LEARNING")
                pred = model(X)[:, 0]

                unscaled_pred = dataset.unnormalize(pred, **dataset.y_kwargs_transform)
                unscaled_y = dataset.unnormalize(y, **dataset.y_kwargs_transform)

                err_pred = (unscaled_pred - unscaled_y).abs().mean().item()
                print(f"Average abs error validation: {err_pred:.2f}")

                traces.append(dict(cat=cat, err=err_pred, learning="after"))

            fig, ax = plt.subplots()
            ax.set_title(f"Loss")
            ax.plot(hist_loss)
            plt.savefig(f"{fig_folder}/hist_loss_feature_identification_cat{cat_index}_fold{fold}.pdf")
            plt.close()
            print()

    os.makedirs("traces", exist_ok=True)
    pd.DataFrame(traces).to_csv("traces/traces.csv")


if __name__ == '__main__':
    main()
