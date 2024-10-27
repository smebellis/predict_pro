import plotly.graph_objects as go
import pandas as pd
import json
import os

# If you have a custom helper function, ensure it's correctly implemented.
# from utils.helper import read_csv_with_progress

# Step 1: Read the Taxi Data
df = pd.read_csv(
    "/home/smebellis/ece5831_final_project/processed_data/post_processing_clustered.csv",
    nrows=10000,
)

# Calculate the average latitude and longitude to center the map
average_lat = df["Lat"].mean()
average_long = df["Long"].mean()

# Step 2: Initialize the Plotly Figure
fig = go.Figure()

# Step 3: Load Districts Data from JSON
with open(
    "/home/smebellis/ece5831_final_project/data/porto_districts.json", "r"
) as FILE:
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
    # Step 4.1: Add labels for each district in the center of the district
    # center_lat = (coords["lower_lat"] + coords["upper_lat"]) / 2
    # center_long = (coords["left_long"] + coords["right_long"]) / 2

    # Add a trace for the label (using mode='text' to display the name of the district)
    # fig.add_trace(
    #     go.Scattermapbox(
    #         lat=[center_lat],
    #         lon=[center_long],
    #         mode="text",
    #         text=[region],
    #         textfont=dict(size=14, color="black"),  # Customize font size and color
    #         hoverinfo="skip",  # Skip hover info for the label
    #     )
    # )

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

# Optional: Add Taxi Data Points (e.g., as Scattermapbox Points)
# Uncomment the following block if you want to visualize taxi locations

# fig.add_trace(
#     go.Scattermapbox(
#         lon=df["Long"],
#         lat=df["Lat"],
#         mode="markers",
#         marker=dict(size=5, color="red", opacity=0.5),
#         name="Taxi Locations",
#         hoverinfo="none",
#     )
# )


# Step 6: Display the Figure
fig.show()

# Step 7: Save the Figure

root_dir = "/home/smebellis/ece5831_final_project/"

save_dir = os.path.join(root_dir, "images")
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

image = "fig1.png"
image_path = os.path.join(save_dir, image)
fig.write_image(image_path)
