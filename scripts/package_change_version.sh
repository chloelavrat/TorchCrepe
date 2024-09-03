#!/bin/bash

# Define the new version number as an argument
NEW_VERSION="$1"

# Check if a version number was provided
if [ -z "$NEW_VERSION" ]; then
    echo "Usage: $0 <new_version>"
    exit 1
fi

# Path to the pyproject.toml file
PYPROJECT_FILE="pyproject.toml"

# Use sed to replace the version number in the pyproject.toml file
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" "$PYPROJECT_FILE"

# Print a success message
echo "Version updated to $NEW_VERSION in $PYPROJECT_FILE"

# Optional: Remove backup file created by sed
rm "${PYPROJECT_FILE}.bak"
