import os
import time

import numpy as np

import modules.helper as helper


def main():
    config, mode, project_name = helper.get_arguments()
    project_path = f"projects/{project_name}/"
    if mode == "newProject":
        helper.create_new_project(project_name)
    elif mode == "train":
        perform_training(config, project_path)
    elif mode == "train+":
        config["activation_extraction"] = True
        perform_training(config, project_path)
    elif mode == "diagnostics":
        perform_diagnostics(config, project_path)
    elif mode == "plot":
        perform_plotting(project_path, config)
    elif mode == "compress":
        perform_compression(config, project_path)
    elif mode == "decompress":
        perform_decompression(config.save_as_root, config.model_name, project_path)
    elif mode == "info":
        print_info(project_path)


def perform_training(config, project_path):
    (
        train_set_norm,
        test_set_norm,
        number_of_columns,
        normalization_features,
    ) = helper.process(
        config.data_path,
        config.names_path,
        config.custom_norm,
        config.test_size,
        config.energy_conversion,
    )
    train_set_norm = helper.normalize(train_set, config)
    test_set_norm = helper.normalize(test_set, config)

    device = helper.get_device()

    ModelObject = helper.model_init(config.model_name)
    model = ModelObject(
        device=device, n_features=number_of_columns, z_dim=config.latent_space_size
    )

    output_path = project_path + "training/"
    test_data_tensor, reconstructed_data_tensor, trained_model = helper.train(
        model, number_of_columns, train_set_norm, test_set_norm, output_path, config
    )
    test_data = helper.detach(test_data_tensor)
    reconstructed_data = helper.detach(reconstructed_data_tensor)

    print("Un-normalzing...")
    print(np.mean(test_data[0]))
    start = time.time()
    test_data_renorm = helper.renormalize(
        test_data,
        normalization_features[0],
        normalization_features[1],
    )
    reconstructed_data_renorm = helper.renormalize(
        reconstructed_data,
        normalization_features[0],
        normalization_features[1],
    )
    end = time.time()
    print("Un-normalization took:", f"{(end - start) / 60:.3} minutes")

    # helper.to_pickle(test_data_renorm, output_path + "before.pickle")
    # helper.to_pickle(reconstructed_data_renorm, output_path + "after.pickle")
    # helper.to_pickle(full_pre_norm, output_path + "fulldata_energy.pickle")
    np.save(output_path + "before.npy", test_data_renorm)
    np.save(output_path + "after.npy", reconstructed_data_renorm)

    np.save(
        project_path + "compressed_output/normalization_features.npy",
        normalization_features,
    )
    helper.model_saver(trained_model, project_path + "compressed_output/model.pt")

def perform_diagnostics(config, project_path):
    print('TBA')


def perform_plotting(project_path, config):
    output_path = project_path + "plotting/"
    helper.plot(project_path, config)
    helper.loss_plotter(project_path + "training/loss_data.npy", output_path, config)


def perform_compression(config, project_path):
    print("Compressing...")
    start = time.time()
    compressed, data_before, cleared_col_names = helper.compress(
        model_path=project_path + "compressed_output/model.pt",
        config=config,
    )
    # Converting back to numpyarray
    compressed = helper.detach(compressed)
    end = time.time()

    print("Compression took:", f"{(end - start) / 60:.3} minutes")

    np.save(project_path + "compressed_output/compressed.npy", compressed)
    np.save(project_path + "compressed_output/names.npy", cleared_col_names)


def perform_decompression(save_as_root, model_name, project_path):
    print("Decompressing...")
    cleared_col_names = np.load(project_path + "compressed_output/names.npy")
    start = time.time()
    decompressed = helper.decompress(
        model_path=project_path + "compressed_output/model.pt",
        input_path=project_path + "compressed_output/compressed.npy",
        model_name=model_name,
    )

    # Converting back to numpyarray
    decompressed = helper.detach(decompressed)
    normalization_features = np.load(
        project_path + "compressed_output/normalization_features.npy"
    )

    decompressed = helper.renormalize(
        decompressed,
        normalization_features[0],
        normalization_features[1],
    )
    end = time.time()
    print("Decompression took:", f"{(end - start) / 60:.3} minutes")
    np.save(project_path + "decompressed_output/decompressed.npy", decompressed)


def print_info(project_path):
    print(
        "================================== \n Information about your compression \n================================== "
    )

    pre_compression = project_path + "compressed_output/cleandata_pre_comp.pickle"
    compressed = project_path + "compressed_output/compressed.pickle"
    decompressed = project_path + "decompressed_output/decompressed.pickle"

    files = [pre_compression, compressed, decompressed]
    q = []
    for i in range(len(files)):
        q.append(os.stat(files[i]).st_size / (1024 * 1024))

    print(
        f"\nCompressed file is {round(q[1] / q[0], 2) * 100}% the size of the original\n"
    )
    print(f"File size before compression: {round(q[0], 2)} MB")
    print(f"Compressed file size: {round(q[1], 2)} MB")
    print(f"De-compressed file size: {round(q[2], 2),} MB")
    print(f"Compression ratio: {round(q[0] / q[1], 2)}")


if __name__ == "__main__":
    main()
