
# Predict Pro: 

This project utilizes GPS data from taxis in Porto, Portugal, to predict traffic status by combining advanced feature engineering, clustering, and machine learning. A preprocessing pipeline extracts key features like travel time, trip distance, and spatial-temporal encodings, while HDBSCAN clustering generates traffic labels to capture urban congestion dynamics.

At its core is TrafficStatusCNN, a convolutional neural network that processes route images and contextual features to predict traffic conditions. This approach highlights the power of integrating spatial and temporal data for real-world traffic management.

The findings provide actionable insights into traffic patterns, aiding city planners, transportation services, and autonomous vehicle systems. Future work aims to incorporate additional data, expand the dataset, and explore real-time deployment for enhanced traffic prediction.

## Dataset

This dataset captures a full year of taxi trip trajectories in Porto, Portugal, spanning from 01/07/2013 to 30/06/2014. It includes data for all 442 taxis operating under a dispatch central, recorded in a single CSV file named "train.csv". These taxis are equipped with mobile data terminals, allowing for detailed trip tracking.

Each trip is categorized into one of three service types:

    A) Central-based: Dispatched via the taxi central, with an anonymized ID provided when available.
    B) Stand-based: Requested directly at taxi stands.
    C) Non-central-based: Hailed directly on the street.

The dataset comprises one record per completed trip, featuring the following 9 attributes:

    TRIP_ID: Unique identifier for each trip.
    CALL_TYPE: Service request type (A for central dispatch, B for taxi stand, C for street hail).
    ORIGIN_CALL: Anonymized customer ID for central dispatch trips (NULL otherwise).
    ORIGIN_STAND: Starting taxi stand for stand-based trips (NULL otherwise).
    TAXI_ID: Unique identifier for the taxi performing the trip.
    TIMESTAMP: Unix timestamp (seconds) indicating the trip's start time.
    DAYTYPE: Type of day (A for normal days, B for holidays, C for days preceding holidays).
    MISSING_DATA: Indicates if GPS data is incomplete (TRUE/FALSE).
    POLYLINE: Encoded list of GPS coordinates in WGS84 format, with trajectories recorded at 15-second intervals.

[Taxi Trajectory Dataset on Kaggle](https://www.kaggle.com/datasets/crailtap/taxi-trajectory)

[Taxi Trajectory Dataset on Google Drive](https://drive.google.com/file/d/1rmJuNl6tenjid_Kp9roIKP176JKTYSvj/view?usp=drive_link)

## Manually download dataset using Kaggle API

 1. Installation

    Ensure you have Python and pip installed on your system.

    Install the Kaggle API package by running the following command:

`pip install kaggle`

On Mac/Linux, if you encounter issues, use:

`pip install --user kaggle`

If you see a kaggle: command not found error:

Check if your Python binaries are added to your system's PATH.
To find where kaggle is installed, run:

    pip uninstall kaggle 
    
    and note the binary location (e.g., ~/.local/bin on Linux or $PYTHON_HOME/Scripts on Windows).

2. Authentication

    Go to the Account tab on your Kaggle profile.

    Click Create New Token to download the kaggle.json file, which contains your API credentials.

    Move the kaggle.json file to the appropriate directory based on your operating system:

    `Linux/Mac/UNIX: ~/.kaggle/kaggle.json`

    `Windows: C:\Users\<Your-Username>\.kaggle\kaggle.json`

    Ensure the `.kaggle` directory and `kaggle.json` file have appropriate permissions to keep your credentials secure.

    If using the Kaggle CLI, the tool automatically searches for the token in these locations.
    If using the API programmatically, ensure you provide the path to your `kaggle.json` file at runtime.
## Presentation

[Presentation Link](https://google.com)
## Final Report

[Final Report Link](https://drive.google.com/file/d/1XSEGkLgptA1-eePTCwSZxg2GHmGxwIjz/view?usp=drive_link)
## Demo Video

[Demo Video](https://youtu.be/cffwHUfAOmo)

## Getting Started

Prerequisites

Before getting started with predict_pro, ensure your runtime environment meets the following requirements:

- **Python 3.7 or higher**: Ensure Python is installed on your system. You can download it from [Python's official website](https://www.python.org/downloads/).

### Clone the Repository

```bash
git clone https://github.com/smebellis/predict_pro

```

Navigate to the project directory:

`❯ cd predict_pro`

Install the project dependencies:

Using pip  

`❯ pip install -r requirements.txt`

Change branches

`> git checkout demo`

1. Run preprocessing pipeline:
- `python scripts/run_preprocessing_pipeline.py`
2. Run Clustering pipeline:
- `python scripts/run_clustering_pipeline.py
3. Run Feature Engineering pipeline
- `python scripts/run_feature_engineering_pipeline.py`
4. Train the model
- `python src/train.py`

Alternatively
1. `open final-project.ipynb`
2. `execute cells in oder`

## Authors

- [Ryan Ellis](https://github.com/smebellis)
- [Ahmad Alamery](https://github.com/ahalamer)


## License

[MIT](https://choosealicense.com/licenses/mit/)

