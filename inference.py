import numpy as np
import deeptrack as dt
import pandas as pd
import tifffile
import argparse
from pathlib import Path


def get_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tiff_file",
        type=str,
        help="Name of tiff file, will be added to ./data/input",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Path to output directory, will be added to ./data/output",
    )
    parser.add_argument(
        "--video_startframe",
        type=int,
        default=0,
        help="Number of frames to skip from the tiff file",
    )
    parser.add_argument(
        "--video_endframe",
        type=int,
        default=None,
        help="Last frame to include from the tiff file, leave empty to use all frames",
    )
    parser.add_argument(
        "--loadstar_path",
        type=str,
        default=r"./weights/loadstar/size_20_AffineAdd/size_20_AffineAdd",
        help="Path to loadstar weights",
    )
    parser.add_argument(
        "--magik_path",
        type=str,
        default=r"./weights/magik/MAGIK_MP_MPN.h5",
        help="Path to magik weights",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.99,
        help="Alpha parameter for loadstar",
    )
    parser.add_argument(
        "--cuttoff",
        type=float,
        default=0.1,
        help="Cutoff parameter for loadstar",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.03,
        help="Radius parameter for magik",
    )
    parser.add_argument(
        "--nframes",
        type=int,
        default=3,
        help="Number of frames parameter for magik",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Threshold parameter for magik",
    )
    args = parser.parse_args()

    return args


def load_and_normalize_images(path, video_startframe, video_endframe):
    """
    Load and normalize images from a TIFF file.

    Args:
        path (str): Path to the TIFF file.
        video_startframe (int): Number of frames to skip from the TIFF file.
        video_cutoff (int): Number of frames to include from the TIFF file.

    Returns:
        numpy.ndarray: Normalized images.
    """
    # Load images
    images = tifffile.imread(path)

    # Normalize
    images = images - np.mean(images)
    images = images / np.std(images, axis=(0, 1, 2), keepdims=True) / 3

    if video_endframe:
        images = images[video_startframe:video_endframe]
    else:
        images = images[video_startframe:]

    return images


def load_models(loadstar_path, magik_path):
    """
    Load LodeSTAR and MAGIK models.

    Args:
        loadstar_path (str): Path to LodeSTAR weights.
        magik_path (str): Path to MAGIK weights.

    Returns:
        tuple: Tuple containing the loaded LodeSTAR and MAGIK models.
    """
    loadSTAR = dt.models.LodeSTAR(input_shape=(None, None, 1))
    loadSTAR.load_weights(loadstar_path)

    # MAGIK
    magik = dt.models.gnns.MAGIK(
        dense_layer_dimensions=(
            64,
            96,
        ),  # number of features in each dense encoder layer
        base_layer_dimensions=(
            96,
            96,
            96,
        ),  # Latent dimension throughout the message passing layers
        number_of_node_features=2,  # Number of node features in the graphs
        number_of_edge_features=1,  # Number of edge features in the graphs
        number_of_edge_outputs=1,  # Number of predicted features
        edge_output_activation="sigmoid",  # Activation function for the output layer
        output_type="edges",  # Output type. Either "edges", "nodes", or "graph"
        graph_block="MPN",
    )
    magik.load_weights(magik_path)

    return loadSTAR, magik


def detect(model, images, alpha, cutoff):
    """
    Detect objects in images using the given model.

    Args:
        model: The detection model.
        images (numpy.ndarray): Input images.
        alpha (float): Alpha parameter for detection.
        cutoff (float): Cutoff parameter for detection.

    Returns:
        pandas.DataFrame: Detected objects.
    """
    detections_all = model.predict_and_detect(
        images, alpha=alpha, beta=1 - alpha, cutoff=cutoff
    )

    detections_all_new = []
    for detection in detections_all:
        detections_all_new.append(
            np.delete(
                detection,
                np.where(
                    (detection[:, 0] < 5)
                    | (detection[:, 1] < 5)
                    | (detection[:, 1] > 121)
                    | (detection[:, 0] > 75)
                )[0],
                axis=0,
            )
        )

    detections_all = detections_all_new

    # Make detections into Pandas DataFrame
    size = 0
    for detections in detections_all:
        size += detections.shape[0]

    detection_array = np.zeros((size, 6))

    traversed = 0
    for i, detections in enumerate(detections_all):
        detection_array[traversed : traversed + detections.shape[0], 0] = i
        detection_array[traversed : traversed + detections.shape[0], 4:] = detections
        traversed += detections.shape[0]

    detection_df = pd.DataFrame(
        detection_array,
        columns=["frame", "set", "label", "solution", "centroid-1", "centroid-0"],
    ).astype(
        {
            "set": "int32",
            "frame": "int32",
            "label": "int32",
            "solution": "int32",
            "centroid-1": "float32",
            "centroid-0": "float32",
        }
    )
    detection_df["centroid-1"] /= images.shape[1]
    detection_df["centroid-0"] /= images.shape[2]

    del detections_all

    return detection_df


def predict_trajectories(model, detections, radius, nframes, threshold):
    """
    Predict trajectories using the given model and detections.

    Args:
        model: The trajectory prediction model.
        detections (pandas.DataFrame): Detected objects.
        radius (float): Radius parameter for trajectory prediction.
        nframes (int): Number of frames parameter for trajectory prediction.
        threshold (int): Trajectory length threshold parameter for trajectory prediction.

    Returns:
        pandas.DataFrame: Predicted trajectories.
    """
    variables = dt.DummyFeature(
        radius=radius,
        output_type="edges",
        nofframes=nframes,  # time window to associate nodes (in frames)
    )

    pred, gt, scores, graph = dt.models.gnns.get_predictions(
        detections, ["centroid"], model, **variables.properties()
    )
    edges_df, nodes, _ = dt.models.gnns.df_from_results(pred, gt, scores, graph)

    # Get trajectories from results
    trajs = dt.models.gnns.to_trajectories(edges_df=edges_df)
    trajs = list(filter(lambda t: len(t) > threshold, trajs))
    trajs = [sorted(t) for t in trajs]

    object_index = np.zeros((nodes.shape[0]))
    for i, traj in enumerate(trajs):
        object_index[traj] = i + 1

    nodes_df = pd.DataFrame(
        {
            "frame": nodes[:, 0],
            "x": nodes[:, 1],
            "y": nodes[:, 2],
            "entity": object_index,
        }
    )
    nodes_df = nodes_df[nodes_df["entity"] != 0]

    return nodes_df


def save_detections(detection_df, path):
    """
    Save the detected objects to a CSV file.

    Args:
        detection_df (pandas.DataFrame): Detected objects.
        path (str): Path to save the CSV file.
    """
    detection_df.drop(columns=["set", "label", "solution"], inplace=True)
    detection_df.to_csv(path, index=False)


def save_trajectories(traj_df, path):
    """
    Save the predicted trajectories to a CSV file.

    Args:
        traj_df (pandas.DataFrame): Predicted trajectories.
        path (str): Path to save the CSV file.
    """
    traj_df.to_csv(path, index=False)


if __name__ == "__main__":
    args = get_args()

    if args.output:
        output_name = args.output + "_"

    images = load_and_normalize_images(
        Path(f"./data/input/{args.tiff_file}"),
        args.video_startframe,
        args.video_endframe,
    )
    loadstar, magik = load_models(Path(args.loadstar_path), Path(args.magik_path))

    detections_df = detect(loadstar, images, args.alpha, args.cuttoff)
    nodes_df = predict_trajectories(
        magik, detections_df, args.radius, args.nframes, args.threshold
    )

    # save_detections(detections_df, Path(f"./data/output/{args.output}detections.csv"))
    save_trajectories(nodes_df, Path(f"./data/output/{args.output}trajectories.csv"))
