# https://colab.research.google.com/drive/1uguA4lluzmi9uuLWxsy2XVE3NtvppNyM#scrollTo=E05AxlVkGu_t
import json
import os
import matplotlib
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchcontrib
# https://github.com/Bjarten/early-stopping-pytorch
from pytorch_lightning.loggers import NeptuneLogger
import config
from pytorchtools import EarlyStopping


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

    # counts of evaluated subjects per age

    age_grouped = df.groupby('real_age').size().reset_index(name='counts')

    age_grouped.to_csv('output/evaluated_count_per_age.csv')

    df_group = df.groupby(['real_age']).mean()

    df_group.to_csv('output/performance_per_age.csv')


def prep_dataset(some_df):
    x_tensor = torch.FloatTensor(list(some_df.face_vector))
    y_tensor = torch.FloatTensor(list(some_df.age)).resize_((x_tensor.shape[0], 1))
    return x_tensor, y_tensor


def get_data(json_path):

    # implement to get fair amount of data per age bins

    with open(json_path) as fh:
        v = json.load(fh)

    unique_names = set(map(lambda _x: _x.split('_')[0], list(v.keys())))
    names = list(map(lambda _x: _x.split('_')[0], list(v.keys())))
    ages = list(map(lambda _x: float(_x.split('_')[1]), list(v.keys())))

    vecs = list(map(lambda _x: v[_x], list(v.keys())))

    df = pd.DataFrame({'name': names, 'age': ages, 'face_vector': vecs})

    fig_histogram, ax = plt.subplots()
    df.hist(ax=ax)
    fig_histogram.savefig('output/histogram.png')

    age_grouped = df.groupby('age').size().reset_index(name='counts')

    age_grouped.to_csv('output/count_per_age.csv')

    unique_names = list(unique_names)

    train_cutoff = int(len(unique_names) * 0.8)
    train_names = unique_names[0:train_cutoff]
    train_names.sort()

    valid_test = unique_names[train_cutoff:-1]

    valid_cutoff = int(len(valid_test) * 0.7)

    valid_names = valid_test[0:valid_cutoff]
    valid_names.sort()

    test_names = valid_test[valid_cutoff:-1]
    test_names.sort()

    print(f'Training on {train_names}')
    print(f'Validating on {valid_names}')
    print(f'Testing on {test_names}')

    df_train = df[df.name.isin(train_names)].copy()
    df_valid = df[df.name.isin(valid_names)].copy()
    df_test = df[df.name.isin(test_names)].copy()

    print(f'We have {df_train.shape[0]}'
          f' training samples, {df_valid.shape[0]}'
          f' validation samples, and {df_test.shape[0]}'
          f' testing samples.')

    x_train, y_train = prep_dataset(df_train)
    x_valid, y_valid = prep_dataset(df_valid)
    x_test, y_test = prep_dataset(df_test)

    # print(f'Loaded {x_train.shape[0]} samples (face vector, age)')
    print(f'Target vector has dimensions {tuple(y_train.shape)}')

    return df, x_train, y_train, x_valid, y_valid, x_test, y_test


def get_model(input_dimension, number_layers, division_factor):
    # D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    D_in, H, D_out = input_dimension, 512, 1

    # Use the nn package to define our model as a sequence of layers. nn.Sequential
    # is a Module which contains other Modules, and applies them in sequence to
    # produce its output. Each Linear Module computes output from input using a
    # linear function, and holds internal Tensors for its weight and bias.

    # modify to make dynamic

    # number_layers = 3
    # division_factor = 2  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    layers = []

    for _ in range(number_layers):
        layers.append(torch.nn.Linear(D_in, H))

        # torch.nn.LeakyReLU()

        layers.append(torch.nn.ReLU())

        D_in = H

        if _ != number_layers - 1:

            H = int(H / division_factor)

            if H == 0:
                H = 1

    layers.append(torch.nn.Linear(H, D_out))

    model = torch.nn.Sequential(
        *layers
    )

    print(model)

    return model


def train_model(model, batch_size, patience, n_epochs, x, y, x_valid, y_valid, learning_rate, experiment_id):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    checkpoint_path = 'checkpoints/{}/checkpoint.pt'.format(experiment_id)

    if not os.path.exists('checkpoints/{}'.format(experiment_id)):
        os.mkdir('checkpoints/{}'.format(experiment_id))

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

    _loss = 0
    val_loss = 0

    criterion = torch.nn.MSELoss()

    base_opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torchcontrib.optim.SWA(base_opt, swa_start=10, swa_freq=5, swa_lr=0.05)

    for epoch in range(epochs):
        model.train()

        for t in range(0, x.shape[0] - batch_size, batch_size):

            # Zero the gradients before running the backward pass.

            optimizer.zero_grad()

            y_pred = model(x[t:t + batch_size, :])

            # Compute and print loss. We pass Tensors containing the predicted and true
            # values of y, and the loss function returns a Tensor containing the
            # loss.
            _loss = criterion(y_pred, y[t:t + batch_size])

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            _loss.backward()

            optimizer.step()

            # print('epoch {}, loss {}'.format(epoch, _loss.item()))

        train_losses.append(_loss.data)

        # validate

        model.eval()

        for v in range(0, x_valid.shape[0] - batch_size, batch_size):

            optimizer.zero_grad()

            y_pred = model(x_valid[v:v + batch_size, :])

            # Compute and print loss. We pass Tensors containing the predicted and true
            # values of y, and the loss function returns a Tensor containing the
            # loss.
            val_loss = criterion(y_pred, y_valid[v:v + batch_size])

        valid_losses.append(val_loss.data)

        # print training/validation statistics
        # calculate average loss over an epoch
        _train_loss = np.average(train_losses)
        _valid_loss = np.average(valid_losses)
        avg_train_losses.append(_train_loss)
        avg_valid_losses.append(_valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {_train_loss:.5f} ' +
                     f'valid_loss: {_valid_loss:.5f}')

        print(print_msg)

        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decreased,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(_valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load(checkpoint_path))

    return model, avg_train_losses, avg_valid_losses


if __name__ == '__main__':

    output_path = config.FACE_VECTOR_PATH

    data_frame, train_x, train_y, valid_x, valid_y, test_x, test_y = get_data(output_path)

    number_layers = 3
    # division_factor = 2  # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    regression_model = get_model(len(data_frame.face_vector[0]), number_layers=number_layers, division_factor=2)

    loss_list = []
    val_loss_list = []
    loss = 0

    batch_size = 64
    epochs = 1000
    patience = 50
    learning_rate = 1e-4
    SWA = {
        'swa_start': 10,
        'swa_freq': 5,
        'swa_lr': 0.05
    }

    experiment_name = 'visage face_vector (revised)'

    if config.LOG_RESULTS:

        matplotlib.use('agg')

        neptune_logger = NeptuneLogger(
            api_key=config.API_KEY,
            project_name="4nd4/sandbox",
            experiment_name=experiment_name,
            close_after_fit=False,
            params={
                "epochs": epochs,
                "patience": patience,
                "number_layers": number_layers,
                "batch_size": batch_size,
                "optimizer": {'SWA': SWA},
                "learning_rate": learning_rate,
                "json_path": output_path
            },
            tags=['regression', 'SWA', 'facenet', 'simple-monitor', 'visage']

        )

        experiment_id = neptune_logger.version

    else:
        neptune_logger = None
        experiment_id = 'DEBUG'

    regression_model, train_loss, valid_loss = train_model(model=regression_model,
                                                           batch_size=batch_size,
                                                           patience=patience,
                                                           n_epochs=epochs,
                                                           x=train_x, y=train_y,
                                                           x_valid=valid_x, y_valid=valid_y,
                                                           learning_rate=learning_rate,
                                                           experiment_id=experiment_id
                                                           )

    y_pred_train = regression_model(train_x)
    train_mae = calc_mae(train_y, y_pred_train)

    print(f'Mean Average Error on training set = {train_mae:.2f}')

    y_pred_valid = regression_model(valid_x)
    valid_mae = calc_mae(valid_y, y_pred_valid)

    print(f'Mean Average Error on validation set = {valid_mae:.2f}')

    y_pred_test = regression_model(test_x)
    test_mae = calc_mae(test_y, y_pred_test)

    print(f'Mean Average Error on test set = {test_mae:.2f}')

    print('creating performance per age file')

    calc_mae_class(test_y, y_pred_test)

    # visualize the loss as the network trained

    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    if config.LOG_RESULTS:

        neptune_logger.experiment.log_metric('mae_train', train_mae)
        neptune_logger.experiment.log_metric('mae_valid', valid_mae)
        neptune_logger.experiment.log_metric('mae_test', test_mae)
        neptune_logger.experiment.log_image('graph', fig)
        neptune_logger.experiment.log_artifact('checkpoints/{}/checkpoint.pt'.format(experiment_id))
        neptune_logger.experiment.log_artifact('output/performance_per_age.csv')
        neptune_logger.experiment.log_artifact('output/histogram.png')
        neptune_logger.experiment.log_artifact('output/count_per_age.csv')
        neptune_logger.experiment.log_artifact('output/evaluated_count_per_age.csv')
        neptune_logger.experiment.stop()
    else:
        plt.show()

    fig.savefig('output/loss_plot.png', bbox_inches='tight')
