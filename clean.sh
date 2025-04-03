#!/bin/bash

echo ""
echo "=================== Starting Cleanup ============================"
echo ""

# ----------------------------------------------------------------
# Step 1: Delete Virtual Environment
VENV_PATH="$(pwd)/BaseEnv"
if [ -d "$VENV_PATH" ]; then
    echo "Deleting virtual environment at $VENV_PATH..."
    rm -rf "$VENV_PATH"
    if [ $? -eq 0 ]; then
        echo "Virtual environment deleted successfully."
    else
        echo "Error deleting virtual environment."
        exit 1
    fi
else
    echo "Virtual environment not found."
fi
echo ""

# ----------------------------------------------------------------
# Step 2: Uninstall Python 3.10.11 using pyenv
PYTHON_VERSION="3.10.11"
if pyenv versions | grep -q "$PYTHON_VERSION"; then
    echo "Uninstalling Python $PYTHON_VERSION using pyenv..."
    pyenv uninstall -f "$PYTHON_VERSION"
    if [ $? -eq 0 ]; then
        echo "Python $PYTHON_VERSION uninstalled successfully."
    else
        echo "Error uninstalling Python $PYTHON_VERSION."
        exit 1
    fi
else
    echo "Python $PYTHON_VERSION is not installed."
fi
echo ""

# ----------------------------------------------------------------
# Step 3: Remove pyenv installation
if [ -d "$HOME/.pyenv" ]; then
    echo "Removing pyenv installation at $HOME/.pyenv..."
    rm -rf "$HOME/.pyenv"
    if [ $? -eq 0 ]; then
        echo "pyenv uninstalled successfully."
    else
        echo "Error removing pyenv installation."
        exit 1
    fi
else
    echo "pyenv is not installed."
fi
echo ""

# ----------------------------------------------------------------
# Step 4: Remove pyenv configuration from ~/.bashrc
BASHRC_FILE="$HOME/.bashrc"
if grep -q 'export PYENV_ROOT="$HOME/.pyenv"' "$BASHRC_FILE"; then
    echo "Removing pyenv configuration from $BASHRC_FILE..."
    sed -i '/export PYENV_ROOT="$HOME\/.pyenv"/d' "$BASHRC_FILE"
    sed -i '/export PATH="$PYENV_ROOT\/bin:$PYENV_ROOT\/shims:$PATH"/d' "$BASHRC_FILE"
    sed -i '/eval "$(pyenv init --path)"/d' "$BASHRC_FILE"
    sed -i '/eval "$(pyenv virtualenv-init -)"/d' "$BASHRC_FILE"
    echo "pyenv configuration removed from $BASHRC_FILE."
else
    echo "No pyenv configuration found in $BASHRC_FILE."
fi
echo ""

echo "=================== Cleanup Complete! ============================"
echo ""
read -p "Press any key to exit..." -n1 -s
echo ""
