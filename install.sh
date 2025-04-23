#!/bin/bash

SOURCE_DIR="src"
TARGET_DIR="/usr/local/bin/app.face.detector"

echo "Installing dependencies"
chmod +x install.dependencies.sh
./install.dependencies.sh

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory '$SOURCE_DIR' not found."
    exit 1
fi

chmod +x "$SOURCE_DIR/face.detector"

echo "Creating target directory $TARGET_DIR"
sudo mkdir -p "$TARGET_DIR"

echo "Copying files from $SOURCE_DIR to $TARGET_DIR"
sudo cp -rf "$SOURCE_DIR"/* "$TARGET_DIR/"

#This way it is you can instantly run the app
sudo mv "$TARGET_DIR"/face.detector /usr/local/bin/
sudo chmod +x /usr/local/bin/face.detector

echo "Installation complete. The application can now be run using 'face.detector'"
