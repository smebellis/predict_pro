import plotly.graph_objects as go
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from torch.utils.tensorboard import SummaryWriter
from TrafficStatusCNN import TrafficStatusCNN
from torchviz import make_dot
import torch


def plot_district_map(
    data_path, json_path, output_dir, output_filename="fig1.png", nrows=10000
):
    # Step 1: Read the Taxi Data
    df = pd.read_csv(data_path, nrows=nrows)

    # Calculate the average latitude and longitude to center the map
    average_lat = df["Lat"].mean()
    average_long = df["Long"].mean()

    # Step 2: Initialize the Plotly Figure
    fig = go.Figure()

    # Step 3: Load Districts Data from JSON
    with open(json_path, "r") as FILE:
        data = json.load(FILE)

    # Step 4: Iterate Through Each District and Add to the Figure
    for region, coords in data.items():
        # Define the latitude and longitude of the rectangle's corners
        lats = [
            coords["lower_lat"],
            coords["upper_lat"],
            coords["upper_lat"],
            coords["lower_lat"],
            coords["lower_lat"],  # Close the loop
        ]
        lons = [
            coords["right_long"],
            coords["right_long"],
            coords["left_long"],
            coords["left_long"],
            coords["right_long"],  # Close the loop
        ]

        # Add each district as a separate Scattermapbox trace
        fig.add_trace(
            go.Scattermapbox(
                name=region,
                lon=lons,
                lat=lats,
                mode="lines",
                fill="toself",
                line=dict(width=2, color="blue"),
                fillcolor="rgba(135, 206, 250, 0.3)",  # LightSkyBlue with transparency
                hoverinfo="text",
                text=region,  # Display the region name on hover
            )
        )

    # Step 5: Update the Layout of the Map
    fig.update_layout(
        title="Porto Districts",
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=average_lat, lon=average_long),
            zoom=12,
        ),
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        showlegend=False,  # Hide legend if not needed
    )

    # Step 6: Save the Figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    image_path = os.path.join(output_dir, output_filename)
    fig.write_image(image_path)

    # Step 7: Display the Figure
    fig.show()


def visualize_with_tensorboard(
    model, images, additional_features, log_dir="logs/traffic_status_cnn"
):
    """
    Visualize a PyTorch model using TensorBoard.

    Args:
        model (nn.Module): The PyTorch model to visualize.
        images (torch.Tensor): Sample input tensor representing images.
        additional_features (torch.Tensor): Sample input tensor representing additional features.
        log_dir (str): Directory to store the TensorBoard logs.
    """
    # Use TensorBoard's SummaryWriter to log the model graph
    writer = SummaryWriter(log_dir)

    # Log the model graph
    writer.add_graph(model, (images, additional_features))

    writer.close()


def visualize_with_torchviz(
    model, images, additional_features, output_filename="traffic_status_cnn"
):
    """
    Visualize a PyTorch model using torchviz.

    Args:
        model (nn.Module): The PyTorch model to visualize.
        images (torch.Tensor): Sample input tensor representing images.
        additional_features (torch.Tensor): Sample input tensor representing additional features.
        output_filename (str): Name of the output file for the visualization graph (without extension).
    """
    # Perform a forward pass to generate the output
    output = model(images, additional_features)

    # Visualize the computational graph using torchviz
    graph = make_dot(output, params=dict(model.named_parameters()))
    graph.render(output_filename, format="png")


def visualize_clusters_status(
    df: pd.DataFrame,
    cluster_column: str,
    lat_column: str = "START_LAT",
    long_column: str = "START_LONG",
    traffic_status_column: str = "TRAFFIC_STATUS",
) -> None:
    """
    Visualize clusters using a scatter plot of geographical coordinates, including traffic status labels.

    Args:
        df (pd.DataFrame): The dataframe containing cluster, coordinate, and traffic status data.
        cluster_column (str): The name of the column containing cluster labels.
        lat_column (str): The name of the column containing latitude coordinates.
        long_column (str): The name of the column containing longitude coordinates.
        traffic_status_column (str): The name of the column containing traffic status labels.

    Returns:
        None: Displays a scatter plot of the clusters.
    """
    # Ensure required columns are in the DataFrame
    if not all(
        col in df.columns
        for col in [cluster_column, lat_column, long_column, traffic_status_column]
    ):
        raise ValueError(
            "Dataframe must contain specified cluster, latitude, longitude, and traffic status columns."
        )

    # Set up the plot
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")

    # Plotting the clusters with traffic status as style
    scatter = sns.scatterplot(
        x=df[long_column],
        y=df[lat_column],
        # hue=df[cluster_column],
        style=df[traffic_status_column],
        palette="viridis",
        alpha=0.6,
        edgecolor="k",
        s=60,
    )

    # Plot details
    plt.title("Cluster Visualization of Geographical Coordinates with Traffic Status")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Adjust legend
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(
        handles=handles,
        labels=labels,
        title="Cluster and Traffic Status",
        loc="best",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.show()


def visualize_clusters(
    df: pd.DataFrame,
    cluster_column: str,
    lat_column: str = "START_LAT",
    long_column: str = "START_LONG",
    traffic_status_column: str = "TRAFFIC_STATUS",
) -> None:
    """
    Visualize clusters using a scatter plot of geographical coordinates, including traffic status labels.

    Args:
        df (pd.DataFrame): The dataframe containing cluster, coordinate, and traffic status data.
        cluster_column (str): The name of the column containing cluster labels.
        lat_column (str): The name of the column containing latitude coordinates.
        long_column (str): The name of the column containing longitude coordinates.
        traffic_status_column (str): The name of the column containing traffic status labels.

    Returns:
        None: Displays a scatter plot of the clusters.
    """
    # Ensure required columns are in the DataFrame
    if not all(
        col in df.columns
        for col in [cluster_column, lat_column, long_column, traffic_status_column]
    ):
        raise ValueError(
            "Dataframe must contain specified cluster, latitude, longitude, and traffic status columns."
        )

    # Extract data for plotting
    clusters = df[cluster_column]
    latitudes = df[lat_column]
    longitudes = df[long_column]
    traffic_status = df[traffic_status_column]

    # Set up the plot
    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid")
    sns.scatterplot(
        x=longitudes,
        y=latitudes,
        hue=clusters,
        style=traffic_status,
        palette="viridis",
        alpha=0.6,
        edgecolor="k",
        s=60,
    )

    # Plot details
    plt.title("Cluster Visualization of Geographical Coordinates with Traffic Status")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(
        title="Cluster and Traffic Status",
        loc="best",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.0,
    )
    plt.tight_layout()
    plt.show()


def visualize_clusters_on_map(
    df: pd.DataFrame,
    cluster_column: str,
    lat_column: str = "START_LAT",
    long_column: str = "START_LONG",
    traffic_status_column: str = "TRAFFIC_STATUS",
) -> None:
    """
    Visualize clusters on a geographical map using Folium, including traffic status labels.

    Args:
        df (pd.DataFrame): The dataframe containing cluster, coordinate, and traffic status data.
        cluster_column (str): The name of the column containing cluster labels.
        lat_column (str): The name of the column containing latitude coordinates.
        long_column (str): The name of the column containing longitude coordinates.
        traffic_status_column (str): The name of the column containing traffic status labels.

    Returns:
        None: Displays an interactive map with the clusters.
    """
    # Ensure required columns are in the DataFrame
    if not all(
        col in df.columns
        for col in [cluster_column, lat_column, long_column, traffic_status_column]
    ):
        raise ValueError(
            "Dataframe must contain specified cluster, latitude, longitude, and traffic status columns."
        )

    # Create a base map centered around the mean coordinates
    mean_lat = df[lat_column].mean()
    mean_long = df[long_column].mean()
    base_map = folium.Map(location=[mean_lat, mean_long], zoom_start=12)

    # Define colors for different clusters
    cluster_colors = sns.color_palette(
        "viridis", n_colors=df[cluster_column].nunique()
    ).as_hex()
    cluster_color_mapping = {
        cluster: cluster_colors[i]
        for i, cluster in enumerate(df[cluster_column].unique())
    }

    # Add markers for each point in the dataframe
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row[lat_column], row[long_column]],
            radius=5,
            color=cluster_color_mapping[row[cluster_column]],
            fill=True,
            fill_color=cluster_color_mapping[row[cluster_column]],
            fill_opacity=0.6,
            popup=folium.Popup(
                f"Cluster: {row[cluster_column]}<br>Traffic Status: {row[traffic_status_column]}",
                parse_html=True,
            ),
        ).add_to(base_map)

    # Display the map
    base_map.save("clusters_map.html")
    print("Map saved as clusters_map.html. Open this file to view the map.")


# Step 7: Save the Figure
if __name__ == "__main__":
    df = pd.read_csv(
        "/home/smebellis/ece5831_final_project/processed_data/clustered_dataset.csv"
    )
    # df = df.sample(n=500, random_state=42)
    visualize_clusters_status(df, cluster_column="CLUSTER")
    # visualize_clusters_on_map(df, cluster_column="CLUSTER")
    breakpoint()
