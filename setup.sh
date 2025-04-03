#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo ""
echo "=================== Starting Setup ============================"
echo ""

# Detect WSL and remove pyenv-win if present
if grep -qi microsoft /proc/version && [ -d "$HOME/.pyenv" ]; then
    if [ -d "$HOME/.pyenv/pyenv-win" ]; then
        echo "Detected pyenv-win in WSL. Removing..."
        rm -rf "$HOME/.pyenv"
        echo "pyenv-win removed."
    fi
fi

# ----------------------------------------------------------------
# Step 1: Install pyenv if not already installed
echo "Checking for pyenv..."
if ! command -v pyenv &> /dev/null; then
    echo "Installing pyenv..."
    curl https://pyenv.run | bash

    # Set up pyenv for the current shell session
    export PYENV_ROOT="$HOME/.pyenv"
    export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
    eval "$(pyenv init --path)"
    eval "$(pyenv virtualenv-init -)"
    echo "pyenv installed successfully!"
    echo ""

    if ! grep -q 'export PYENV_ROOT="$HOME/.pyenv"' ~/.bashrc; then
        echo "Appending pyenv configuration to ~/.bashrc..."
        {
            echo 'export PYENV_ROOT="$HOME/.pyenv"'
            echo 'export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"'
            echo 'eval "$(pyenv init --path)"'
            echo 'eval "$(pyenv virtualenv-init -)"'
        } >> ~/.bashrc
    fi

    # Reload shell
    source ~/.bashrc
else
    echo "pyenv is already installed."
fi

echo "------------------ pyenv setup complete --------------------"
echo ""

# ----------------------------------------------------------------
# Step 2: Install Python 3.10.11 using pyenv
echo "Installing required build dependencies..."
sudo apt update && sudo apt install -y \
    build-essential libbz2-dev libreadline-dev libssl-dev libffi-dev zlib1g-dev \
    liblzma-dev libsqlite3-dev tk-dev libncurses5-dev libncursesw5-dev \
    xz-utils wget curl git
echo "Build dependencies installed."

PYTHON_VERSION="3.10.11"
if ! pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Installing Python $PYTHON_VERSION using pyenv..."
    pyenv install "$PYTHON_VERSION"
    echo "Python $PYTHON_VERSION installed successfully."
else
    echo "Python $PYTHON_VERSION is already installed."
fi

echo "Setting global Python version to $PYTHON_VERSION..."
pyenv global "$PYTHON_VERSION"
echo "Python version set to $PYTHON_VERSION."
echo "---------------- Python Setup Step Complete ----------------------"
echo ""

# ----------------------------------------------------------------
# Step 3: Create a virtual environment
VENV_PATH="$(pwd)/BaseEnv"
if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment in $VENV_PATH..."
    python -m venv "$VENV_PATH"
    echo "Virtual environment created successfully."
else
    echo "Virtual environment already exists at $VENV_PATH."
fi

echo "---------------- Virtual Environment Creation Complete -----------"
echo ""

# ----------------------------------------------------------------
# Step 4: Activate virtual environment and install required modules
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "Upgrading pip..."
python -m pip install --upgrade pip

REQ_FILE="$(pwd)/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "Installing required modules from requirements.txt..."
    python -m pip install -r "$REQ_FILE"
    echo "All required modules have been installed."
else
    echo "requirements.txt not found in the current directory."
fi

echo "=================== Setup Complete! ============================"
echo ""

# ----------------------------------------------------------------
# Step 5: Run InsightEX
echo "=================== Launching AI Module ======================"
echo ""
"$VENV_PATH/bin/python" -m InsightEX || echo "Error: Could not start InsightEX"

echo ""
read -p "Press any key to exit..."
