#!/bin/bash

# Create directory if it doesn't exist
mkdir -p ./dataset/HumanML3D

echo "Downloading CLIP embeddings and dataset files..."

# Your Google Drive file ID for CLIP embeddings
FILE_ID="1BBMlDFfAGAnpjXx_DwNxTKCFow15NmHk"
FILE_NAME="clip_embeddings.zip"

# Download using gdown
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

# Extract the zip file
echo "Extracting files..."
unzip $FILE_NAME -d ./temp_extract/

# Move files from temp_clip to HumanML3D
echo "Moving files to correct directory..."
cp -r ./temp_extract/temp_clip/* ./dataset/HumanML3D/

# Clean up
rm -rf ./temp_extract
rm $FILE_NAME

echo "Download complete! Dataset files are in the ./dataset/HumanML3D/ directory."