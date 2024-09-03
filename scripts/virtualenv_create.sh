#!/bin/bash

# Define color codes
SUCCESS='\033[0;32m'
RESET='\033[0m' # Reset color

# Function to create a virtual environment
create_virtual_env() {
    echo -e "${SUCCESS}Setting up virtual environment...${RESET}"
    python3 -m venv .venv > /dev/null
}

# Function to activate the virtual environment
activate_virtual_env() {
    echo -e "${SUCCESS}Activating virtual environment...${RESET}"
    source .venv/bin/activate > /dev/null
}

# Function to upgrade pip
upgrade_pip() {
    echo -e "${SUCCESS}Upgrading pip...${RESET}"
    pip install --upgrade pip > /dev/null
}

# Function to install dependencies
install_dependencies() {
    echo -e "${SUCCESS}Installing dependencies from requirements file...${RESET}"
    pip install -r requirements.txt
}

# Function to display completion message
display_completion() {
    echo -e "${SUCCESS}Setup complete!${RESET}"
}

# Main script execution
create_virtual_env
activate_virtual_env
upgrade_pip
install_dependencies
display_completion