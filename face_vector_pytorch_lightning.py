# https://colab.research.google.com/drive/1uguA4lluzmi9uuLWxsy2XVE3NtvppNyM#scrollTo=E05AxlVkGu_t
import json
import sys
import matplotlib
import os
import pandas as pd
import torch
import numpy as np
import torchcontrib
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.neptune import NeptuneLogger
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
import config
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import random

log_folder = 'logs'

if config.LOG_RESULTS:
    matplotlib.use('agg')

gettrace = getattr(sys, 'gettrace', None)
num_workers = config.NUM_WORKERS
random_seed = random.randint(0, 10000)

# detect if debug mode

if gettrace is None:
    num_workers = 0

pin_memory = False

train_json = 'output/out_train_visage_instagram_augmented.json'
test_json = 'output/out_test_visage_instagram_augmented.json'


# We'll also want to calculate a human-friendly metric
def calc_mae(y, y_hat):
    diff = torch.abs(y - y_hat)
    return torch.mean(diff)


def calc_mae_class(y, y_hat):
    diff = torch.abs(y - y_hat)

    np_y = y.numpy()
    np_y = np_y.astype(int)
    np_diff = diff.detach().numpy()

    csv_text = np.column_stack([np_y, np_diff])

    df = pd.DataFrame(csv_text, columns=['real_age', 'difference'])

    df_group = df.groupby(['real_age']).mean()

    df_group.to_csv('{}/performance_per_age.csv'.format(log_folder))

    print('performance per age saved....')


def prep_dataset(some_df):
    x_tensor = torch.FloatTensor(list(some_df.face_vector))
    y_tensor = torch.FloatTensor(list(some_df.age)).resize_((x_tensor.shape[0], 1))
    return x_tensor, y_tensor


def get_data_old(json_path):
    with open(json_path) as fh:
        v = json.load(fh)

    keys = list(map(lambda _x: _x.split('_')[0], list(v.keys())))
    ages = list(map(lambda _x: float(_x.split('_')[1]), list(v.keys())))
    vecs = list(map(lambda _x: v[_x], list(v.keys())))

    df = pd.DataFrame({'key': keys, 'age': ages, 'face_vector': vecs})

    # rework this

    # get balanced cut

    train_set = pd.DataFrame()
    test_valid_set = pd.DataFrame()
    valid_set = pd.DataFrame()
    test_set = pd.DataFrame()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=random_seed)
    for train_index, test_valid_index in split.split(df, df.age):
        train_set = df.iloc[train_index]
        test_valid_set = df.iloc[test_valid_index]

    split2 = StratifiedShuffleSplit(n_splits=1, test_size=0.6, random_state=random_seed)
    for test_index, valid_index in split2.split(test_valid_set, test_valid_set.age):
        test_set = test_valid_set.iloc[test_index]
        valid_set = test_valid_set.iloc[valid_index]

    print(f'We have {train_set.shape[0]}'
          f' training samples, {valid_set.shape[0]}'
          f' validation samples, and {test_set.shape[0]}'
          f' testing samples.')

    x_train, y_train = prep_dataset(train_set)
    x_valid, y_valid = prep_dataset(valid_set)
    x_test, y_test = prep_dataset(test_set)

    print(f'Target vector has dimensions {tuple(y_train.shape)}')

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def get_data(json_path_train, json_path_test):

    with open(json_path_train) as fh_train:
        v_train = json.load(fh_train)

    keys = list(map(lambda _x: _x.split('_')[0], list(v_train.keys())))
    ages = list(map(lambda _x: float(_x.split('_')[1]), list(v_train.keys())))
    vecs = list(map(lambda _x: v_train[_x], list(v_train.keys())))

    df_train = pd.DataFrame({'key': keys, 'age': ages, 'face_vector': vecs})

    age_grouped = df_train.groupby('age').size().reset_index(name='counts')
    age_grouped.to_csv('{}/count_per_age_train.csv'.format(log_folder))

    with open(json_path_test) as fh_test:
        v_test = json.load(fh_test)

    keys = list(map(lambda _x: _x.split('_')[0], list(v_test.keys())))
    ages = list(map(lambda _x: float(_x.split('_')[1]), list(v_test.keys())))
    vecs = list(map(lambda _x: v_test[_x], list(v_test.keys())))

    df_test = pd.DataFrame({'key': keys, 'age': ages, 'face_vector': vecs})

    age_grouped = df_test.groupby('age').size().reset_index(name='counts')
    age_grouped.to_csv('{}/count_per_age_test.csv'.format(log_folder))

    train_set = df_train
    test_valid_set = df_test
    valid_set = pd.DataFrame()
    test_set = pd.DataFrame()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.95, random_state=random_seed)
    for test_index, valid_index in split.split(df_test, df_test.age):
        test_set = test_valid_set.iloc[test_index]
        valid_set = test_valid_set.iloc[valid_index]

    print(f'We have {train_set.shape[0]}'
          f' training samples, {valid_set.shape[0]}'
          f' validation samples, and {test_set.shape[0]}'
          f' testing samples.')

    x_train, y_train = prep_dataset(train_set)
    x_valid, y_valid = prep_dataset(valid_set)
    x_test, y_test = prep_dataset(test_set)

    print(f'Target vector has dimensions {tuple(y_train.shape)}')

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


(train_x, train_y), (val_x, val_y), (test_x, test_y) = get_data(json_path_train=train_json, json_path_test=test_json)


class Vec2UAgeSystem(pl.LightningModule):

    def __init__(self):
        super(Vec2UAgeSystem, self).__init__()

        self.model = get_model(input_dimension=512, number_layers=3, division_factor=2)
        self.criterion = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        loss = loss.unsqueeze(dim=-1)

        mae = calc_mae(y, y_hat)

        tensorboard_logs = {'loss': loss, 'mae': mae}

        return {
            'loss': loss,
            'mae': mae,
            'log': tensorboard_logs
        }

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        val_loss = self.criterion(y_hat, y)
        val_loss = val_loss.unsqueeze(dim=-1)

        val_mae = calc_mae(y, y_hat)

        return {
            'val_loss': val_loss,
            'val_mae': val_mae
        }

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        avg_mae = torch.stack([x['val_mae'] for x in outputs]).mean()

        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_mae': avg_mae
        }

        # log debugging images like histogram of losses
        fig = plt.figure()
        losses = np.stack([x['val_loss'].numpy() for x in outputs])
        plt.hist(losses)

        if config.LOG_RESULTS:
            self.logger.experiment.log_image('loss_histograms', fig)
        else:
            # plt.show()
            pass

        plt.close(fig)

        return {
            'avg_val_loss': avg_loss,
            'avg_mae_loss': avg_mae,
            'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)

        test_loss = self.criterion(y_hat, y)
        test_loss = test_loss.unsqueeze(dim=-1)

        test_mae = calc_mae(y, y_hat)

        return {
            'test_loss': test_loss,
            'test_mae': test_mae
        }

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_mae = torch.stack([x['test_mae'] for x in outputs]).mean()

        tensorboard_logs = {
            'test_loss': avg_loss,
            'test_mae': avg_mae
        }

        return {
            'avg_test_loss': avg_loss,
            'avg_test_mae': avg_mae,
            'log': tensorboard_logs
        }

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)

        base_opt = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        optimizer = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)

        return optimizer

    def train_dataloader(self):
        # REQUIRED

        data_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=batch_size,
            pin_memory=pin_memory,
            # shuffle=True,
            num_workers=num_workers

        )

        return data_loader

    def val_dataloader(self):
        # OPTIONAL

        data_loader = DataLoader(
            TensorDataset(val_x, val_y),
            batch_size=batch_size,
            pin_memory=pin_memory,
            # shuffle=True,
            num_workers=num_workers
        )

        return data_loader

    def test_dataloader(self):
        # OPTIONAL

        data_loader = DataLoader(
            TensorDataset(test_x, test_y),
            batch_size=batch_size,
            pin_memory=pin_memory,
            # shuffle=True,
            num_workers=num_workers
        )

        return data_loader


def get_data_test(json_path):
    with open(json_path) as fh:
        v = json.load(fh)

    keys = list(map(lambda _x: _x.split('_')[0], list(v.keys())))
    ages = list(map(lambda _x: float(_x.split('_')[1]), list(v.keys())))

    vecs = list(map(lambda _x: v[_x], list(v.keys())))

    df = pd.DataFrame({'key': keys, 'age': ages, 'face_vector': vecs})

    x, y = prep_dataset(df)

    return x, y


def stats(json_path):
    with open(json_path) as fh:
        v = json.load(fh)

    names = list(map(lambda _x: _x.split('_')[0], list(v.keys())))
    ages = list(map(lambda _x: float(_x.split('_')[1]), list(v.keys())))

    vecs = list(map(lambda _x: v[_x], list(v.keys())))

    df = pd.DataFrame({'name': names, 'age': ages, 'face_vector': vecs})

    fig_histogram, ax = plt.subplots()
    df.hist(ax=ax)
    fig_histogram.savefig('output/histogram.png')

    age_grouped = df.groupby('age').size().reset_index(name='counts')
    age_grouped.to_csv('output/count_per_age.csv')


def get_model(input_dimension, number_layers, division_factor):
    # D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    dimension_input, hidden_dimension, dimension_output = input_dimension, 512, 1

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.

    # modify to make dynamic

    # number_layers = 3
    # division_factor = 2  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    layers = []

    for _ in range(number_layers):
        layers.append(torch.nn.Linear(dimension_input, hidden_dimension))

        # torch.nn.LeakyReLU()

        layers.append(torch.nn.ReLU())

        dimension_input = hidden_dimension

        if _ != number_layers - 1:

            hidden_dimension = int(hidden_dimension / division_factor)

            if hidden_dimension == 0:
                hidden_dimension = 1

    layers.append(torch.nn.Linear(hidden_dimension, dimension_output))

    model = torch.nn.Sequential(
        *layers
    )

    print(model)

    return model


if __name__ == '__main__':

    batch_size = config.BATCH_SIZE
    epochs = config.MAXIMUM_EPOCHS
    patience = 20
    learning_rate = 1e-4
    nb_layers = 3
    early_stop = True
    SWA = {
        'swa_start': 10,
        'swa_freq': 5,
        'swa_lr': 0.05
    }

    # stats(output_path)

    experiment_name = 'pl_face_vector'

    tags = ['regression', 'SWA', 'facenet', 'pytorch-lightning']

    if not early_stop:
        patience = 0
    else:
        tags.append('early-stopping')

    if not config.LOG_RESULTS:
        experiment_name = 'DEBUG'
        CHECKPOINTS_DIR = 'checkpoints/{}/'.format(experiment_name)
        neptune_logger = None

    else:
        neptune_logger = NeptuneLogger(
            api_key=config.API_KEY,
            project_name="4nd4/sandbox",
            experiment_name=experiment_name,
            close_after_fit=False,
            params={
                "epochs": epochs,
                "patience": patience,
                "number_layers": nb_layers,
                "batch_size": batch_size,
                "optimizer": {'SWA': SWA},
                "learning_rate": learning_rate,
                "json_path": [train_json, test_json],
                "early_stop": early_stop,
                "random_seed": random_seed
            },
            tags=tags

        )

        CHECKPOINTS_DIR = 'checkpoints/{}/'.format(neptune_logger.version)

    if not os.path.exists(CHECKPOINTS_DIR):
        os.mkdir(CHECKPOINTS_DIR)
        print('checkpoint directory created', CHECKPOINTS_DIR)

    seed_everything(random_seed)

    pl_model = Vec2UAgeSystem()

    checkpoint_callback = ModelCheckpoint(
        filepath='{}/working'.format(CHECKPOINTS_DIR),
        verbose=True,
        monitor='avg_val_loss',
        mode='min',
    )

    if early_stop:
        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,
            patience=patience,
            verbose=True,
            mode='min'
        )
    else:
        early_stop_callback = None

    trainer = Trainer(
        deterministic=True,
        max_epochs=epochs,
        checkpoint_callback=checkpoint_callback,
        logger=neptune_logger,
        early_stop_callback=early_stop_callback
    )

    trainer.fit(pl_model)
    trainer.test(pl_model)

    # Save checkpoints folder

    y_pred_train = pl_model(train_x)
    train_mae = calc_mae(train_y, y_pred_train)

    print(f'Mean Average Error on training set = {train_mae:.2f}')

    y_pred_valid = pl_model(val_x)
    valid_mae = calc_mae(val_y, y_pred_valid)

    print(f'Mean Average Error on validation set = {valid_mae:.2f}')

    y_pred_test = pl_model(test_x)
    test_mae = calc_mae(test_y, y_pred_test)

    print(f'Mean Average Error on test set = {test_mae:.2f}')

    print('creating performance per age file')

    calc_mae_class(test_y, y_pred_test)

    if config.LOG_RESULTS:

        matplotlib.use('agg')

        neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR)
        # neptune_logger.experiment.log_artifact('{}/performance_per_age.csv'.format(log_folder))
        neptune_logger.experiment.log_artifact('{}/count_per_age_train.csv'.format(log_folder))
        neptune_logger.experiment.log_artifact('{}/count_per_age_test.csv'.format(log_folder))
        # You can stop the experiment
        neptune_logger.experiment.stop()
    else:
        # draw results

        plt.show()
