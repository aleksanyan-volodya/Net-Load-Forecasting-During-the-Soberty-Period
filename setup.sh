#!/usr/bin/env bash

set -e  # error handling

echo "Checking Python 3..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 is not installed or not in PATH"
    exit 1
fi

echo "Python version:"
python3 --version

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment already exists."
fi

echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found. Skipping dependency installation."
fi

echo "Environment setup complete."
echo "To activate later, run:"
echo "source .venv/bin/activate"
