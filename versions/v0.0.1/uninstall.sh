#!/bin/bash

TARGET_DIR="/usr/local/bin/app.face.detector"

echo "Removing application directory $TARGET_DIR"
sudo rm -rf "$TARGET_DIR"
sudo rm -rf /usr/local/bin/face.detector

echo "Uninstallation complete. The application has been removed."

