import modules.models as models
import modules.training as training
import modules.plotting as plotting
import modules.data_processing as data_processing
import argparse
import json
import pickle
import torch
import pandas
import numpy

def get_arguments():
    parser = argparse.ArgumentParser(
                    prog = "baler.py",
                    description =   '''Baler is a machine learning based compression tool for big data.\n
                                    Baler has three running modes:\n
                                    \t1. Derivation: Using a configuration file and a "small" input dataset, Baler derives a machine learning model optimized to compress and decompress your data.\n
                                    \t2. Compression: Using a previously derived model and a large input dataset, Baler compresses your data and outputs a smaller compressed file.\n
                                    \t3. Decompression: Using a previously compressed file as input and a model, Baler decompresses your data into a larger file.''',
                    epilog = 'Enjoy!')
    parser.add_argument('--config', type=str, required=False, help='Path to config file')
    parser.add_argument('--model', type=str, required=False, help='Path to previously derived machinelearning model')
    parser.add_argument('--input', type=str, required=True, help='Path to input data set for compression')
    parser.add_argument('--output', type=str, required=True, help='Path of output data')
    parser.add_argument('--mode', type=str, required=True, help='train, compress, decompress, plot, info ')

    args = parser.parse_args()
    if args.config: config = data_processing.import_config(args.config)
    else: config = args.config
    return args.input, args.output, args.model, config, args.mode

def to_pickle(data, path):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle)

def model_init(config):
    # This is used when we don't have saved model parameters.
    ModelObject = data_processing.initialise_model(config=config)
    return ModelObject

def data_loader(data_path,config):
    return data_processing.load_data(data_path,config)

def numpy_to_tensor(data):
    if isinstance(data, pandas.DataFrame):
        data = data.to_numpy()
    
    return torch.from_numpy(data)

def normalize(data,config):
    data = numpy.apply_along_axis(data_processing.normalize,axis=0,arr=data,config=config)
    df = data_processing.numpy_to_df(data)
    return df

def process(data_path, config):
    df = data_processing.load_data(data_path,config)
    df = data_processing.clean_data(df,config)
    normalization_features = data_processing.find_minmax(df)
    df = normalize(df,config)
    train_set, test_set = data_processing.split(df, test_size=config["test_size"], random_state=1)
    number_of_columns = len(data_processing.get_columns(df))
    assert number_of_columns == config["number_of_columns"], f"The number of columns of dataframe is {number_of_columns}, config states {config['number_of_columns']}."
    return train_set, test_set, number_of_columns, normalization_features

def renormalize(data,true_min_list,feature_range_list,config):
    return data_processing.renormalize_func(data,true_min_list,feature_range_list)

def train(model,number_of_columns,train_set,test_set,project_path,config):
    return training.train(model, number_of_columns, train_set, test_set, project_path, config)

def plot(test_data, reconstructed_data):
    plotting.plot(test_data, reconstructed_data)

def loss_plotter(path_to_loss_data,output_path):
    return plotting.loss_plot(path_to_loss_data,output_path)

def model_loader(model_path):
    return data_processing.load_model(model_path)

def model_saver(model,model_path):
    return data_processing.save_model(model,model_path)

def detach(tensor):
    return tensor.detach().numpy()

def compress(number_of_columns,model_path,input_path,config):
    # Initialise and load the model correctly.
    ModelObject = data_processing.initialise_model(config=config)
    model = data_processing.load_model(ModelObject, model_path = model_path, n_features=number_of_columns, z_dim = config["latent_space_size"])

    # Give the encoding function the correct input as tensor
    data = data_loader(input_path, config)
    data = data_processing.clean_data(data,config)
    data_before = numpy.array(data)

    data = normalize(data,config)
    data_tensor = numpy_to_tensor(data)

    compressed = model.encode(data_tensor)
    return compressed, data_before

def decompress(number_of_columns,model_path, input_path, config):
    # Initialise and load the model correctly.
    ModelObject = data_processing.initialise_model(config=config)
    model = data_processing.load_model(ModelObject,
                                       model_path=model_path,
                                       n_features=number_of_columns,
                                       z_dim = config["latent_space_size"])

    # Load the data & convert to tensor
    data = data_loader(input_path, config)
    data_tensor = numpy_to_tensor(data)

    decompressed = model.decode(data_tensor)
    return decompressed 
