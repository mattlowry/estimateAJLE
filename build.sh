#!/bin/bash

# Exit on any error
set -e

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt

# Clean previous builds
rm -rf build dist *.spec

# Run PyInstaller
pyinstaller --onefile --windowed Starter.py

echo "Build complete! Your app is in dist/Starter"
