#!/bin/bash

SUCCESS='\033[0;32m'
RESET='\033[0m'

# Function to print messages in green
print_success() {
    echo -e "${SUCCESS}$1${RESET}"
}

# Function to activate the Python virtual environment
activate_venv() {
    print_success "Activating Python virtual environment..."
    source .venv/bin/activate > /dev/null
}

# Function to check if a Python package is installed
check_package() {
    if ! python -c "import $1" 2>/dev/null; then
        echo "Error: $1 package is missing."
        exit 1
    fi
}

# Function to upgrade pip
upgrade_pip() {
    print_success "Upgrading pip to the latest version..."
    pip install --upgrade pip > /dev/null
}

# Function to install a Python package
install_package() {
    local package=$1
    print_success "Installing ${package} package..."
    pip install "$package" > /dev/null
}

# Function to generate API documentation using lazydocs
generate_docs() {
    print_success "Creating API documentation with lazydocs..."
    lazydocs \
        --output-path="./docs/api-docs" \
        --overview-file="README.md" \
        --src-base-url="https://github.com/" \
        APOLLO_LIBRARY > /dev/null
}

# Main script execution
activate_venv
check_package "pip"
upgrade_pip
install_package "lazydocs"
generate_docs
