import argparse
import os
import pickle
import sys

import numpy
import pandas
import torch

from modules import training, plotting, data_processing


def get_arguments():
    parser = argparse.ArgumentParser(
        prog="baler.py",
        description="""Baler is a machine learning based compression tool for big data.\n
Baler has three running modes:\n
\t1. Derivation: Using a configuration file and a "small" input dataset, Baler derives a machine learning model optimized to compress and decompress your data.\n
\t2. Compression: Using a previously derived model and a large input dataset, Baler compresses your data and outputs a smaller compressed file.\n
\t3. Decompression: Using a previously compressed file as input and a model, Baler decompresses your data into a larger file.""",
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
        project_path = f"projects/{args.project}/"
        #config_path = f"projects/{args.project}/config.py"
        sys.path.append(project_path)
        import configClass
        config = configClass.Configuration()
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
    with open(os.path.join(project_path, "configClass.py"), "w") as f:
        print(project_path)
        f.write(create_default_config())
    for directory in required_directories:
        os.makedirs(os.path.join(project_path, directory))


def create_default_config() -> str:
    return f"""
class Configuration(object):
    def __init__(self):
        self.input_path = "data/firstProject/cms_data.root"

        self.epochs = 5
        self.early_stopping = True
        self.lr_scheduler = False
        self.patience = 100
        self.min_delta = 0
        self.model_name = "george_SAE"
        self.custom_norm = False
        self.l1 = True
        self.reg_param = 0.001
        self.RHO = 0.05
        self.lr = 0.001
        self.batch_size = 512
        self.save_as_root = True
        self.cleared_col_names = ["pt","eta","phi","m","EmEnergy","HadEnergy","InvisEnergy","AuxilEnergy"]
        self.test_size = 0.15
        self.Branch = "Events"
        self.Collection = "recoGenJets_slimmedGenJets__PAT."
        self.Objects = "recoGenJets_slimmedGenJets__PAT.obj"
        self.number_of_columns = 8
        self.latent_space_size = 4
        self.dropped_variables = [
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fX",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fY",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.vertex_.fCoordinates.fZ",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.qx3_",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.pdgId_",
            "recoGenJets_slimmedGenJets__PAT.obj.m_state.status_",
            "recoGenJets_slimmedGenJets__PAT.obj.mJetArea",
            "recoGenJets_slimmedGenJets__PAT.obj.mPileupEnergy",
            "recoGenJets_slimmedGenJets__PAT.obj.mPassNumber"
        ]
"""


def to_pickle(data, path):
    with open(path, "wb") as handle:
        pickle.dump(data, handle)

def from_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def model_init(config):
    # This is used when we don't have saved model parameters.
    ModelObject = data_processing.initialise_model(config=config)
    return ModelObject


def data_loader(data_path, config):
    return data_processing.load_data(data_path, config)


def numpy_to_tensor(data):
    if isinstance(data, pandas.DataFrame):
        data = data.to_numpy()

    return torch.from_numpy(data)


def normalize(data, config):
    data = numpy.apply_along_axis(
        data_processing.normalize, axis=0, arr=data, config=config
    )
    df = data_processing.numpy_to_df(data, config)
    return df


def process(data_path, config):
    df = data_processing.load_data(data_path, config)
    #df = data_processing.clean_data(df, config)
    normalization_features = data_processing.find_minmax(df)
    config.cleared_col_names = data_processing.get_columns(df)
    number_of_columns = len(data_processing.get_columns(df))
    df = normalize(df, config)

    print("\nNumber of input variables",len(list(df.columns)))
    print("List of input variables",list(df.columns))
    print("\n")

    train_set, test_set = data_processing.split(
        df, test_size=config.test_size, random_state=1
    )
    #assert (
    #    number_of_columns == config.number_of_columns
    #), f"The number of columns of dataframe is {number_of_columns}, config states {config.number_of_columns}."
    return train_set, test_set, number_of_columns, normalization_features


def renormalize(data, true_min_list, feature_range_list):
    return data_processing.renormalize_func(data, true_min_list, feature_range_list)


def train(model, number_of_columns, train_set, test_set, project_path, config):
    return training.train(
        model, number_of_columns, train_set, test_set, project_path, config
    )


def plot(project_path):
    plotting.plot(project_path)


def loss_plotter(path_to_loss_data, output_path, config):
    return plotting.loss_plot(path_to_loss_data, output_path, config)


def model_saver(model, model_path):
    return data_processing.save_model(model, model_path)


def detach(tensor):
    return tensor.cpu().detach().numpy()


def compress(model_path, input_path, config):
    # Give the encoding function the correct input as tensor
    data = data_loader(input_path, config)
    config.cleared_col_names = data_processing.get_columns(data)
    number_of_columns = len(data_processing.get_columns(data))
    try:
        config.latent_space_size = int(number_of_columns//config.compression_ratio)
        config.number_of_columns = number_of_columns
    except AttributeError:
        print(config.latent_space_size,config.number_of_columns)
        assert(number_of_columns==config.number_of_columns)
    data_before = numpy.array(data)
    data = normalize(data, config)
    
    # Initialise and load the model correctly.
    ModelObject = data_processing.initialise_model(config=config)
    model = data_processing.load_model(
        ModelObject,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=config.latent_space_size,
    )
    data_tensor = numpy_to_tensor(data).to(model.device)
    compressed = model.encode(data_tensor)
    return compressed, data_before


def decompress(model_path, input_path, config):

    # Load the data & convert to tensor
    data = data_loader(input_path, config)
    latent_space_size = len(data[0])
    #number_of_columns=8
    modelDict = torch.load(str(model_path))
    number_of_columns = len(modelDict[list(modelDict.keys())[-1]])


    # Initialise and load the model correctly.
    ModelObject = data_processing.initialise_model(config=config)
    model = data_processing.load_model(
        ModelObject,
        model_path=model_path,
        n_features=number_of_columns,
        z_dim=latent_space_size,
    )
    data_tensor = numpy_to_tensor(data).to(model.device)
    decompressed = model.decode(data_tensor)
    return decompressed


def to_root(data_path, config, save_path):
    # if '.pickle' in data_path[-8:]:
    if isinstance(data_path, pickle.Pickler):
        df, Names = data_processing.pickle_to_df(file_path=data_path, config=config)
        return data_processing.df_to_root(df, Names, save_path)
    elif isinstance(data_path, pandas.DataFrame):
        return data_processing.df_to_root(
            data_path, col_names=data_path.columns(), save_path=save_path
        )
    elif isinstance(data_path, numpy.ndarray):
        df = data_processing.numpy_to_df(data_path, config)
        df_names = df.columns
        return data_processing.df_to_root(df, col_names=df_names, save_path=save_path)


def get_device():
    device = None
    if torch.cuda.is_available():
        dev = "cuda:0"
        device = torch.device(dev)
    else:
        dev = "cpu"
        device = torch.device(dev)
    return device
