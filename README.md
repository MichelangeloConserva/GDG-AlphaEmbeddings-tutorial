# Classifying Land Cover with Google Earth Engine & PyTorch

This Colab notebook demonstrates a complete workflow for land cover classification, bridging the gap between cloud-based geospatial data processing (Google Earth Engine) and local deep learning (PyTorch).

## What This Notebook Does

The goal is to train a simple Multi-Layer Perceptron (MLP) neural network to predict land cover classes (like 'forest', 'cropland', 'water', etc.) using a powerful, pre-computed satellite data embedding called AlphaEarth.

Instead of training a model on raw, complex satellite imagery (like Sentinel-2), we will:

- Use AlphaEarth embeddings as our input features. These are 64-band "embeddings" from Google that have already learned to represent complex geospatial patterns, making our model's job much easier.

- Use ESA WorldCover as our "ground truth" labels. This is a high-resolution global map of land cover types.

- Train a PyTorch model locally on data sampled from Google Earth Engine (GEE).

- Run inference on a new, unseen area by downloading the AlphaEarth data for that region.

- Export the final prediction as a standard GeoTIFF file, a common format in remote sensing.

- Visualize the result interactively on a map, comparing our model's prediction to the "true" ESA WorldCover map.

## The Workflow at a Glance

This notebook is broken down into these key steps:

- Setup & Authentication: We will install the necessary Python libraries (rasterio, geemap, torch) and authenticate with Google Earth Engine.

- Data Exploration (GEE): We will load and visualize our three key GEE datasets:

  - **Sentinel-2**: This is a high-resolution optical imaging satellite mission from the European Space Agency (ESA). We use it to provide "true color" (Red, Green, Blue) visual context, just like what our eyes see. While we could train a model on this raw data, it's complex and computationally expensive.

  - **AlphaEarth**: This is a powerful, pre-computed dataset from Google. It's a 64-band "embedding" created by training a deep learning model on vast amounts of global satellite imagery. Each band represents a learned, high-level feature (like "texture," "edges," or "vegetation type") rather than just raw light values. Using these embeddings as our features makes our classification task significantly easier.

  - **ESA WorldCover**: This is our "ground truth" or "answer key." It's a global, high-resolution (10m) land cover map produced by the ESA. It classifies every pixel into one of 11 distinct classes, such as 'Tree cover', 'Cropland', 'Built-up', 'Permanent water bodies', etc. Our model will learn to predict these specific classes.

- Training Data Preparation (GEE): We will define a training Area of Interest (AOI), remap the WorldCover labels into a model-friendly format (0, 1, 2...), and randomly sample 10,000 points from our feature and label images.

- Local Model Training (PyTorch):

  - Download the 10,000 sampled points into a pandas DataFrame.

  - Prepare the data using scikit-learn and PyTorch DataLoaders.

  - Define and train a simple MLP network to learn the mapping from AlphaEarth embeddings to land cover labels.

  - Plot the model's training and validation accuracy.

- Inference & Export (PyTorch & Rasterio):

  - Define a separate "test" AOI in a different location.

  - Download the entire AlphaEarth image for this test AOI as a NumPy array.

  - Run the trained PyTorch model on every pixel in this array to generate a prediction map.

  - Save this prediction map as a world_cover_prediction.tif file, adding the necessary geospatial metadata (like coordinates) using the rasterio library.

- Final Visualization (Geemap): We will create a final interactive map to compare our model's saved .tif file side-by-side with the original ESA WorldCover data, allowing us to visually assess our model's performance.

### Important Colab Runtime Note

You will need to authenticate your Google account to use Earth Engine. When you run the "Authenticate & Initialize" cell:

- A popup will appear asking you to "Permit this notebook to access your Google Account".

- Click "Allow".

- A second popup will ask you to choose your Google account.

- Sign in and grant the necessary permissions.

- You must do this once per Colab session.
