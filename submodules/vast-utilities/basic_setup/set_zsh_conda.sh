#!/bin/bash

# Install Zsh
sudo apt update
sudo apt install -y zsh

# Set Zsh as default shell
chsh -s $(which zsh)

# Install Oh My Zsh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

# Initialize conda for zsh
~/miniconda3/bin/conda init zsh

echo "Zsh installation complete."

# Use a subshell to run conda-related commands
(
    # Source the zshrc file to load conda initialization
    source ~/.zshrc

    # Disable conda's auto base environment activation
    conda config --set auto_activate_base false

    # Check for CUDA support and create a conda environment
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA GPU detected. Creating conda environment with CUDA support."
        conda create -n workspace python=3.10 cudatoolkit -y
    else
        echo "No NVIDIA GPU detected. Creating conda environment without CUDA support."
        conda create -n workspace python=3.10 -y
    fi

    # Add conda activation to shell configuration files
    echo "cd /workspace" >> ~/.zshrc

    # Add conda to PATH
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc

    echo "Conda installation complete."
)
