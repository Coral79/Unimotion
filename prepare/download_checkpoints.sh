#!/bin/bash

# Create save directory if it doesn't exist
mkdir -p ./save

echo "Downloading pretrained models..."

# Your Google Drive file ID
FILE_ID="14743qUr1FLPiFMVk9O6d0jYxqbahkPiI"
FILE_NAME="unimotion_models.zip"

# Download using gdown (handles Google Drive's download process)
echo "Downloading from Google Drive..."
if command -v gdown &> /dev/null; then
    gdown --fuzzy https://drive.google.com/uc?id=$FILE_ID
else
    echo "gdown not found. Installing gdown..."
    pip install gdown
    gdown --fuzzy https://drive.google.com/uc?id=$FILE_ID
fi

# Check if download was successful
if [ ! -f "$FILE_NAME" ]; then
    echo "Error: Download failed. Please check your internet connection and try again."
    exit 1
fi

# Extract to save directory
echo "Extracting models to ./save/ directory..."
unzip $FILE_NAME -d ./save/

# Clean up
rm $FILE_NAME

echo "Download complete! Models are in the ./save/unimotion_pca_51_humanml_trans_enc_512/ directory."