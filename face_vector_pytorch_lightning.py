import argparse
import json
import sys
import matplotlib
import os
import pandas as pd
import torch
import numpy as np
import torchcontrib
from pytorch_lightning import seed_everything, Trainer
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

# random_seed = random.randint(0, 10000)
random_seed = 9057

# detect if debug mode

if gettrace is None:
    num_workers = 0

pin_memory = False

dict_optimizers = {
    'SWA': {
        'swa_start': 10,
        'swa_freq': 5,
        'swa_lr': 0.05,
    },

    'SGD': {
        'lr': 'auto'
    },

    'ADAM': {
        'lr': 'auto'
    },

    'ADAGRAD': {
        'lr': 'auto'
    }
}

train_json = 'output/out_train_visage_instagram_augmented.json'
test_json = 'output/out_test_visage_instagram_augmented.json'


def get_args():
    parser = argparse.ArgumentParser(description="This script trains the regression model for age estimation.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--opt", type=str, required=True,
                        help="ADAM, SGD or SWA")
    arguments = parser.parse_args()
    return arguments


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

    train_set = df_train
    test_valid_set = df_test
    valid_set = pd.DataFrame()
    test_set = pd.DataFrame()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.95, random_state=random_seed)
    for test_index, valid_index in split.split(df_test, df_test.age):
        test_set = test_valid_set.iloc[test_index]
        valid_set = test_valid_set.iloc[valid_index]

    age_grouped = test_set.groupby('age').size().reset_index(name='counts')
    age_grouped.to_csv('{}/count_per_age_test.csv'.format(log_folder))

    age_grouped = valid_set.groupby('age').size().reset_index(name='counts')
    age_grouped.to_csv('{}/count_per_age_valid.csv'.format(log_folder))

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

    def __init__(self, lr, batch_size, optimizer_choice):
        super(Vec2UAgeSystem, self).__init__()

        self.model = get_model(input_dimension=512, number_layers=3, division_factor=2)
        self.criterion = torch.nn.MSELoss()
        self.hparams.batch_size = batch_size
        self.hparams.lr = lr
        self.optimizer_choice = optimizer_choice
        # self.criterion = adaptive.AdaptiveLossFunction(1, torch.float32, device='cpu')

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)

        loss = self.criterion(y_hat, y)

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

        print('learning rate is', self.hparams.lr)

        optimizer = None

        if self.optimizer_choice == 'SWA':

            dict_SWA = dict_optimizers[self.optimizer_choice]

            base_opt = torch.optim.SGD(
                self.model.parameters(),
                # lr=dict_SWA.get('lr')
                lr=self.hparams.lr
            )

            optimizer = torchcontrib.optim.SWA(base_opt, swa_start=dict_SWA.get('swa_start'),
                                               swa_freq=dict_SWA.get('swa_freq'), swa_lr=dict_SWA.get('swa_lr'))

        elif self.optimizer_choice == 'SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
        elif self.optimizer_choice == 'ADAM':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        elif self.optimizer_choice == 'ADAGRAD':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True)
        # reduce every epoch (default)

        return [optimizer], [scheduler]

    def train_dataloader(self):
        # REQUIRED

        data_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=self.hparams.batch_size,
            pin_memory=pin_memory,
            shuffle=True,
            num_workers=num_workers)

        return data_loader

    def val_dataloader(self):
        # OPTIONAL

        data_loader = DataLoader(
            TensorDataset(val_x, val_y),
            batch_size=self.hparams.batch_size,
            pin_memory=pin_memory,
            shuffle=False,
            num_workers=num_workers
        )

        return data_loader

    def test_dataloader(self):
        # OPTIONAL

        data_loader = DataLoader(
            TensorDataset(test_x, test_y),
            batch_size=self.hparams.batch_size,
            pin_memory=pin_memory,
            shuffle=False,
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

    # print(model)
    # summary(model, (1, 512))

    return model


if __name__ == '__main__':

    #################################

    # PARAMS
    unique_tag = 'C1'
    epochs = config.MAXIMUM_EPOCHS
    patience = 10
    nb_layers = 3
    learning_rate = 1e-4
    early_stop = True
    project_name = "4nd4/Vec2UAge"
    learning_rate_finder = True if unique_tag == 'C1' else 'B1'

    args = get_args()

    opt_choice = args.opt
    # optimizer_choice = 'SWA'

    if opt_choice is None:
        print('no optimizer chosen')
        exit(0)

    #################################

    # stats(output_path)

    experiment_name = 'pl_face_vector'

    tags = [opt_choice, unique_tag]

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
            project_name=project_name,
            experiment_name=experiment_name,
            close_after_fit=False,
            params={
                "epochs": epochs,
                "patience": patience,
                "number_layers": nb_layers,
                "batch_size": config.BATCH_SIZE,
                "optimizer": opt_choice,
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

    pl_model = Vec2UAgeSystem(lr=learning_rate, batch_size=config.BATCH_SIZE, optimizer_choice=opt_choice)

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
        # auto_scale_batch_size=True,
        deterministic=True,
        max_epochs=epochs,
        checkpoint_callback=checkpoint_callback,
        logger=neptune_logger,
        early_stop_callback=early_stop_callback,
    )

    if learning_rate_finder:

        lr_finder = trainer.lr_find(
            pl_model,
            train_dataloader=pl_model.train_dataloader(),
            val_dataloaders=pl_model.val_dataloader()
        )

        # fig = lr_finder.plot()
        # fig.show()

        suggested_lr = lr_finder.suggestion()

        pl_model.hparams.lr = suggested_lr

        print('lr overridden:', pl_model.hparams.lr)

        neptune_logger.log_metric('learning_rate', suggested_lr)

    trainer.fit(pl_model)

    trainer.test()

    y_pred_test = pl_model(test_x)

    print('creating performance per age file')

    calc_mae_class(test_y, y_pred_test)

    if config.LOG_RESULTS:

        matplotlib.use('agg')

        neptune_logger.experiment.log_artifact(CHECKPOINTS_DIR)
        neptune_logger.experiment.log_artifact('{}/performance_per_age.csv'.format(log_folder))
        neptune_logger.experiment.log_artifact('{}/count_per_age_train.csv'.format(log_folder))
        neptune_logger.experiment.log_artifact('{}/count_per_age_test.csv'.format(log_folder))
        # You can stop the experiment
        neptune_logger.experiment.stop()
    else:
        # draw results

        plt.show()
