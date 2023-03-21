import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn import functional as F

import modules.utils as utils
import modules.helper as helper

import os


# Helper function for activation extraction
def get_activation(layer):
    def hook(model, input, output):
        layer = output.detach()
    return hook

def fit(model, train_dl, train_ds, model_children, regular_param, optimizer, RHO, l1):
    print("### Beginning Training")

    model.train()

    running_loss = 0.0
    counter = 0

    for inputs in tqdm(train_dl):
        counter += 1
        inputs = inputs.to(model.device)
        optimizer.zero_grad()
        reconstructions = model(inputs)
        loss, mse_loss, l1_loss = utils.sparse_loss_function_l1(
            model_children=model_children,
            true_data=inputs,
            reconstructed_data=reconstructions,
            reg_param=regular_param,
            validate=False,
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f"# Finished. Training Loss: {loss:.6f}")
    return epoch_loss, mse_loss, l1_loss, model


def validate(model, test_dl, model_children, reg_param):
    print("### Beginning Validating")

    model.eval()
    counter = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs in tqdm(test_dl):
            counter += 1
            inputs = inputs.to(model.device)
            reconstructions = model(inputs)
            loss = utils.sparse_loss_function_l1(
                model_children=model_children,
                true_data=inputs,
                reconstructed_data=reconstructions,
                reg_param=reg_param,
                validate=True,
            )
            running_loss += loss.item()

    epoch_loss = running_loss / counter
    print(f"# Finished. Validation Loss: {loss:.6f}")
    return epoch_loss


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train(model, variables, train_data, test_data, parent_path, config):
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    g = torch.Generator()
    g.manual_seed(0)

    test_size = config.test_size
    learning_rate = config.lr
    bs = config.batch_size
    reg_param = config.reg_param
    rho = config.RHO
    l1 = config.l1
    epochs = config.epochs
    latent_space_size = config.latent_space_size
    activation_extraction = config.activation_extraction

    model_children = list(model.children())

    # Initialize model with appropriate device
    device = helper.get_device()
    model = model.to(device)

    # Converting data to tensors
    train_ds = torch.tensor(train_data, dtype=torch.float64, device=device)
    valid_ds = torch.tensor(test_data, dtype=torch.float64, device=device)

    # Pushing input data into the torch-DataLoader object and combines into one DataLoaders object (a basic wrapper
    # around several DataLoader objects).
    train_dl = DataLoader(
        train_ds, batch_size=bs, shuffle=False, worker_init_fn=seed_worker, generator=g
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=bs, worker_init_fn=seed_worker, generator=g
    )  # Used to be batch_size = bs * 2

    # Select Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Activate early stopping
    if config.early_stopping:
        early_stopping = utils.EarlyStopping(
            patience=config.patience, min_delta=config.min_delta
        )  # Changes to patience & min_delta can be made in configs

    # Activate LR Scheduler
    if config.lr_scheduler:
        lr_scheduler = utils.LRScheduler(optimizer=optimizer, patience=config.patience)

    # train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    start = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        train_epoch_loss, mse_loss_fit, regularizer_loss_fit, trained_model = fit(
            model=model,
            train_dl=train_dl,
            model_children=model_children,
            optimizer=optimizer,
            RHO=rho,
            regular_param=reg_param,
            l1=l1,
        )

        train_loss.append(train_epoch_loss)

        if test_size:
            val_epoch_loss = validate(
                model=trained_model,
                test_dl=valid_dl,
                model_children=model_children,
                reg_param=reg_param,
            )
            val_loss.append(val_epoch_loss)
        else:
            val_epoch_loss = train_epoch_loss
            val_loss.append(val_epoch_loss)

        if config.lr_scheduler:
            lr_scheduler(val_epoch_loss)
        if config.early_stopping:
            early_stopping(val_epoch_loss)
            if early_stopping.early_stop:
                break

        ## Make-shift implementation to save models & values after 100 epochs:
        save_model_and_data = True
        if save_model_and_data:
            if epoch % 100 == 0:
                path = os.path.join(parent_path, f"model_{epoch}.pt")
                path_data = os.path.join(parent_path, f"after_{epoch}.pickle")
                path_pred = os.path.join(parent_path, f"before_{epoch}.pickle")

                helper.model_saver(model, path)
                data_tensor = torch.tensor(test_data, dtype=torch.float64).to(
                    model.device
                )
                pred_tensor = model(data_tensor)
                data = helper.detach(data_tensor)
                pred = helper.detach(pred_tensor)

                helper.to_pickle(data, path_data)
                helper.to_pickle(pred, path_pred)

    end = time.time()

    print(f"{(end - start) / 60:.3} minutes")
    np.save(parent_path + "loss_data.npy", np.array([train_loss, val_loss]))

    if activation_extraction:
        hooks = model.store_hooks() 

    data_as_tensor = torch.tensor(test_data, dtype=torch.float64)
    data_as_tensor = data_as_tensor.to(trained_model.device)
    pred_as_tensor = trained_model(data_as_tensor)

    if activation_extraction:
        activations = model.get_activations()
        model.detach_hooks(hooks)
    else:
        activations = {}

    return data_as_tensor, pred_as_tensor, trained_model, activations
