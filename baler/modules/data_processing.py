from argparse import Namespace
from matplotlib.pyplot import sca
from matplotlib.style import library
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import json
import pandas as pd
import uproot 
import numpy as np
import torch

from pathlib import Path

def import_config(config_path):
    with open (config_path) as json_config:
        config = json.load(json_config)
    return config

def save_model(model,model_path):
    return torch.save(model.state_dict(),model_path)

def initialise_model(config):
    import modules.models as model_path
    model_name = config["model_name"]
    ModelObject = getattr(model_path, model_name)
    return ModelObject

def load_model(ModelObject,model_path,n_features,z_dim):
    model = ModelObject(n_features,z_dim)

    #Loading the state_dict into the model
    model.load_state_dict(torch.load(str(model_path)),strict=False)
    return model 

def Type_clearing(TTree):
    typenames = TTree.typenames()
    Column_Type = []
    Column_names = []

    # In order to remove non integers or -floats in the TTree, we separate the values and keys
    for keys in typenames:
        Column_Type.append(typenames[keys])
        Column_names.append(keys)
 
    # Checks each value of the typename values to see if it isn't an int or float, and then removes it
    for i in range(len(Column_Type)):
        if Column_Type[i] != 'float[]' and Column_Type[i] != 'int32_t[]':
            #print("Index ",i," was of type ",Typename_list_values[i]," and was deleted from the file")
            del Column_names[i]
    
    # Returns list of column names to use in load_data function
    return Column_names

def numpy_to_df(array,config):
    df = pd.DataFrame(array,columns=cleared_column_names)
    return df

def load_data(data_path,config):
    path = Path(data_path)
    file_extension = path.suffix

    if ".csv" in file_extension:
        data_file = pd.read_csv(data_path,low_memory=False)
    elif ".root" in file_extension:
        tree = uproot.open(data_path)[config["Branch"]][config["Collection"]][config["Objects"]]
        global Names
        Names = Type_clearing(tree)
        data_file = tree.arrays(Names, library="pd")
    elif ".pickle" in file_extension:
        data_file = pd.read_pickle(data_path)
    elif ".hdata_file5" in file_extension:
        data_file = pd.read_pickle(data_path)

    return data_file

def clean_data(df,config):
    df = df.drop(columns=config["dropped_variables"])
    df = df.dropna()
    global cleared_column_names
    cleared_column_names = list(df)
    return df

def find_minmax(data):
    data = np.array(data)
    data = list(data)
    true_max_list = np.apply_along_axis(np.max, axis=0, arr=data)
    true_min_list = np.apply_along_axis(np.min, axis=0, arr=data)

    feature_range_list = true_max_list - true_min_list

    normalization_features = pd.DataFrame({'True min':true_min_list,
                                            'Feature Range': feature_range_list})
    return normalization_features

def normalize(data,config):
    data = np.array(data)
    if  config["custom_norm"] == True:
        pass
    elif config["custom_norm"] == False:
        true_min = np.min(data)
        true_max = np.max(data)
        feature_range = true_max - true_min
        data = [((i - true_min)/feature_range) for i in data]
        data = np.array(data)
    return data

def split(df, test_size,random_state):
    return train_test_split(df, test_size=test_size, random_state=random_state)

def renormalize_std(data,true_min,feature_range):
    data = list(data)
    data = [((i * feature_range) + true_min) for i in data]
    data = np.array(data)
    return data

def renormalize_func(norm_data,min_list,range_list,config):
    norm_data = np.array(norm_data)
    renormalized = [renormalize_std(norm_data,min_list[i],range_list[i]) for i in range(len(min_list))]
    renormalizedFull = [(renormalized[i][:,i]) for i in range(len(renormalized))]
    renormalizedFull = np.array(renormalizedFull).T
    #renormalizedFull = pd.DataFrame(renormalizedFull,columns=config["cleared_column_names"])
    return renormalizedFull

def get_columns(df):
    return list(df.columns)
