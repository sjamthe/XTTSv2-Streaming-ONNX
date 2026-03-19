#!/bin/bash

# Define the target directory
MODEL_DIR="xtts_onnx"
# Replace this with the exact Hugging Face URL you used
HF_REPO_URL="https://huggingface.co/pltobing/XTTSv2-Streaming-ONNX"

# 1. Create the directory if it doesn't exist
mkdir -p $MODEL_DIR

# 2. Shallow clone the Hugging Face repo to a temporary folder
echo "Downloading ONNX models from Hugging Face (This may take a few minutes)..."
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 $HF_REPO_URL temp_models
cd temp_models
git lfs pull
cd ..

# 3. Move the necessary files into the permanent folder
echo "Moving model files into $MODEL_DIR..."
mv temp_models/*.onnx $MODEL_DIR/ 2>/dev/null
mv temp_models/vocab.json $MODEL_DIR/ 2>/dev/null
mv temp_models/mel_stats.npy $MODEL_DIR/ 2>/dev/null
mv temp_models/config.json $MODEL_DIR/ 2>/dev/null

# 4. Clean up the heavy temporary folder
echo "Cleaning up temporary files..."
rm -rf temp_models

echo "✅ Setup Complete! The models are safely placed in $MODEL_DIR."
echo "You can now run 'docker compose up -d' to start the server."
