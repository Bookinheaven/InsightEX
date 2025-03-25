#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Step 1: Install pyenv if not already installed
if ! command -v pyenv &> /dev/null; then
    echo "🔹 Installing pyenv..."

    # Check for required dependencies
    sudo apt update && sudo apt install -y \
        make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev \
        wget curl llvm libncursesw5-dev xz-utils tk-dev \
        libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

    # Install pyenv
    curl https://pyenv.run | bash

    # Update shell configuration
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
    source ~/.bashrc

    echo "✅ pyenv installed successfully!"
else
    echo "✅ pyenv is already installed."
fi

# Step 2: Install Python 3.10.11 if not installed
PYTHON_VERSION="3.10.11"
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "🔹 Installing Python $PYTHON_VERSION using pyenv..."
    pyenv install $PYTHON_VERSION
else
    echo "✅ Python $PYTHON_VERSION is already installed."
fi

# Set the Python version globally
pyenv global $PYTHON_VERSION

# Step 3: Define environment paths
BASE_ENV_DIR="$HOME/.BaseEnv"
VENV_DIR="$BASE_ENV_DIR/.OpenVino"

# Step 4: Create base environment directory if not exists
if [ ! -d "$BASE_ENV_DIR" ]; then
    echo "🔹 Creating base environment directory at $BASE_ENV_DIR..."
    mkdir -p "$BASE_ENV_DIR"
fi

# Step 5: Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "🔹 Creating OpenVINO virtual environment in $VENV_DIR..."
    python -m venv "$VENV_DIR"
else
    echo "✅ Virtual environment already exists. Skipping creation."
fi

# Step 6: Activate the virtual environment
echo "🔹 Activating OpenVINO environment..."
source "$VENV_DIR/bin/activate"

# Step 7: Upgrade pip and install dependencies
echo "🔹 Upgrading pip..."
pip install --upgrade pip

echo "🔹 Installing OpenVINO and other dependencies..."
pip install openvino openvino-dev ultralytics torch opencv-python numpy

# Step 8: Verify installation
echo "🔹 Verifying installation..."
python -c "import openvino; import torch; import cv2; print('✅ OpenVINO Environment Setup Complete!')"

echo "🎉 OpenVINO Environment is set up successfully!"
