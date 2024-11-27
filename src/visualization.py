import plotly.graph_objects as go
import pandas as pd
import json
import os
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


# Step 7: Save the Figure
if __name__ == "__main__":
    # plot_district_map(
    #     data_path="/home/smebellis/ece5831_final_project/processed_data/post_processing_clustered.csv",
    #     json_path="/home/smebellis/ece5831_final_project/data/porto_districts.json",
    #     output_dir="/home/smebellis/ece5831_final_project/images",
    # )

    # Load the tensor from the .pt file
    additional_features_tensor = torch.load(
        "preprocessed_tensors/val_additional_features_tensor.pt", weights_only=True
    )

    images_size = torch.load(
        "preprocessed_tensors/val_route_images_tensor.pt", weights_only=True
    )
    print(f"Shape of images: {images_size.shape}")
    # Instantiate your model
    num_additional_features = additional_features_tensor.shape[1]
    print("Number of additional features:", num_additional_features)
    breakpoint()
    model = TrafficStatusCNN(num_additional_features)

    # Create sample input
    images = torch.randn(2, 1, 64, 64)
    additional_features = torch.randn(2, num_additional_features)

    # Visualize with torchviz
    visualize_with_torchviz(model, images, additional_features)
