import argparse
import importlib
import os
import pickle
from dataclasses import dataclass
import sys

sys.path.append(os.getcwd())
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from modules import training, plotting, data_processing


def get_arguments():
    parser = argparse.ArgumentParser(
        prog="baler.py",
        description=(
            "Baler is a machine learning based compression tool for big data.\n\n"
            "Baler has three running modes:\n\n"
            '\t1. Derivation: Using a configuration file and a "small" input dataset, Baler derives a '
            "machine learning model optimized to compress and decompress your data.\n\n"
            "\t2. Compression: Using a previously derived model and a large input dataset, Baler compresses "
            "your data and outputs a smaller compressed file.\n\n"
            "\t3. Decompression: Using a previously compressed file as input and a model, Baler decompresses "
            "your data into a larger file."
        ),
        epilog="Enjoy!",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=False,
        help="train, compress, decompress, plot, info",
    )
    parser.add_argument(
        "--project", type=str, required=False, help="Name of new project"
    )

    args = parser.parse_args()
    if not args.mode or (args.mode != "newProject" and not args.project):
        parser.print_usage()
        exit(1)
    if args.mode == "newProject":
        config = None
    else:
        config = Config
        importlib.import_module(
            f"projects.{args.project}.{args.project}_config"
        ).set_config(config)
    return config, args.mode, args.project


def create_new_project(project_name: str, base_path: str = "projects") -> None:
    project_path = os.path.join(base_path, project_name)
    if os.path.exists(project_path):
        print(f"The project {project_path} already exists.")
        return

    required_directories = [
        "compressed_output",
        "decompressed_output",
        "plotting",
        "training",
        "model",
    ]
    os.makedirs(project_path)
    with open(os.path.join(project_path, f"{project_name}_config.py"), "w") as f:
        print(project_path)
        f.write(create_default_config(project_name))
    for directory in required_directories:
        os.makedirs(os.path.join(project_path, directory))


@dataclass
class Config:
    input_path: str
    compression_ratio: float
    epochs: int
    early_stopping: bool
    lr_scheduler: bool
    patience: int
    min_delta: int
    model_name: str
    custom_norm: bool
    l1: bool
    reg_param: float
    RHO: float
    lr: float
    batch_size: int
    save_as_root: bool
    test_size: float
    energy_conversion: bool
    data_dimension: int


def create_default_config(project_name: str) -> str:
    return f"""
def set_config(c):
    c.input_path          = "data/{project_name}/{project_name}_data.npz"
    c.compression_ratio   = 2.0
    c.epochs              = 5
    c.energy_conversion = False
    c.early_stopping      = True
    c.lr_scheduler        = False
    c.patience            = 100
    c.min_delta           = 0
    c.model_name          = "george_SAE"
    c.custom_norm         = False
    c.l1                  = True
    c.reg_param             = 0.001
    c.RHO                 = 0.05
    c.lr                  = 0.001
    c.batch_size          = 512
    c.save_as_root        = True
    c.test_size           = 0.15
    c.energy_conversion   = False
    c.data_dimension      = 1
    c.apply_normalization = True
    c.extra_compression   = False

"""


def to_pickle(data, path):
    with open(path, "wb") as handle:
        pickle.dump(data, handle)


def from_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def model_init(model_name):
    # This is used when we don't have saved model parameters.
    model_object = data_processing.initialise_model(model_name)
    return model_object


def numpy_to_tensor(data):
    return torch.from_numpy(data)


def normalize(data, custom_norm):
    data = np.apply_along_axis(
        data_processing.normalize, axis=0, arr=data, custom_norm=custom_norm
    )
    return data


def process(input_path, custom_norm, test_size, energy_conversion, apply_normalization):
    loaded = np.load(input_path)
    data = loaded["data"]
    names = loaded["names"]
    normalization_features = 0

    # TODO Fix this
    # if energy_conversion:
    #     print("Converting mass to energy with eta, pt & mass")
    #     df = convert_mass_to_energy(df, cleared_col_names)
    normalization_features = data_processing.find_minmax(data)
    if apply_normalization:
        print("Normalizing the data...")
        data = normalize(data, custom_norm)
    if not test_size:
        train_set = data
        test_set = train_set
    else:
        train_set, test_set = train_test_split(
            data, test_size=test_size, random_state=1
        )

    return (
        train_set,
        test_set,
        normalization_features,
    )


def renormalize(data, true_min_list, feature_range_list):
    return data_processing.renormalize_func(data, true_min_list, feature_range_list)


def train(model, number_of_columns, train_set, test_set, project_path, config):
    return training.train(
        model, number_of_columns, train_set, test_set, project_path, config
    )


def plot(project_path, config):
    plotting.plot(project_path, config)


def loss_plotter(path_to_loss_data, output_path, config):
    return plotting.loss_plot(path_to_loss_data, output_path, config)


def model_saver(model, model_path):
    return data_processing.save_model(model, model_path)


def detach(tensor):
    return tensor.cpu().detach().numpy()


def compress(model_path, config):
    # Give the encoding function the correct input as tensor
    loaded = np.load(config.input_path)
    data_before = loaded["data"]
    if config.apply_normalization:
        print("Normalizing...")
        data = normalize(data_before, config.custom_norm)
    else:
        data = data_before
    number_of_columns = 0
    try:
        print("compression ratio:", config.compression_ratio)
        if config.data_dimension == 1:
            column_names = np.load(config.input_path)["names"]
            number_of_columns = len(column_names)
            latent_space_size = int(number_of_columns // config.compression_ratio)
            config.number_of_columns = number_of_columns
        elif config.data_dimension == 2:
            data = np.load(config.input_path)["data"]
            number_of_rows = data.shape[1]
            number_of_columns = data.shape[2]
            config.latent_space_size = int(
                (number_of_rows * number_of_columns) // config.compression_ratio
            )
        else:
            raise NameError(
                "Data dimension can only be 1 or 2. Introduced value = "
                + str(config.data_dimension)
            )
    except AttributeError:
        number_of_columns = config.number_of_columns
        latent_space_size = config.latent_space_size
        print(number_of_columns, latent_space_size)

    # Initialise and load the model correctly.
    latent_space_size = config.latent_space_size
    model_object = data_processing.initialise_model(config.model_name)
    model = data_processing.load_model(
        model_object,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=latent_space_size,
    )

    if config.data_dimension == 2:
        data_tensor = (
            torch.from_numpy(data.astype("float32", casting="same_kind"))
            .to(model.device)
            .view(data.shape[0], 1, data.shape[1], data.shape[2])
        )
    elif config.data_dimension == 1:
        data_tensor = torch.from_numpy(data).to(model.device)

    compressed = model.encode(data_tensor)
    return compressed


def decompress(model_path, input_path, model_name):
    # Load the data & convert to tensor
    loaded = np.load(input_path)
    data = loaded["data"]
    names = loaded["names"]
    normalization_features = loaded["normalization_features"]
    latent_space_size = len(data[0])
    model_dict = torch.load(str(model_path))
    number_of_columns = len(model_dict[list(model_dict.keys())[-1]])

    # Initialise and load the model correctly.
    model_object = data_processing.initialise_model(model_name)
    model = data_processing.load_model(
        model_object,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=latent_space_size,
    )

    # Load the data & convert to tensor
    data_tensor = torch.from_numpy(data).to(model.device)

    decompressed = model.decode(data_tensor)
    return decompressed, names, normalization_features


def get_device():
    device = None
    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu"
        device = torch.device(dev)
    return device


def compute_e(mass, eta, pt):
    masspt = pt**2 + mass**2
    cosh = (np.cosh(eta)) ** 2
    total = np.sqrt(masspt * cosh)
    return total


def convert_mass_to_energy(df, col_names):
    # TODO Not used right now

    for i in range(len(col_names)):
        if col_names[i].split(".")[-1] == "pt":
            pt = df.iloc[:, i]

        if col_names[i].split(".")[-1] == "mass_":
            mass = df.iloc[:, i]

            # Store name to rename & replace mass in df:
            mass_name: str = str(col_names[i])

        if col_names[i].split(".")[-1] == "99":
            eta = df.iloc[:, i]

        else:
            print(
                "Can't convert to energy. Please turn off `energy_conversion` in the config to continue"
            )
            exit(1)

    # Compute mass
    energy = compute_e(mass=mass, eta=eta, pt=pt)

    # Get correct new column name
    energy_name = mass_name.replace("mass_", "energy_")

    # Replace mass with energy
    df[mass_name] = energy

    # Replace column name
    df.columns = df.columns.str.replace(mass_name, energy_name, regex=True)
    return df
