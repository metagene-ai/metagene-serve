#!/bin/bash

# Change to home directory on Vast instance
cd /workspace 

# Download the AWS CLI zip file
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

# Install unzip if not already installed
if ! command -v unzip &> /dev/null; then
    echo "Unzip not found. Installing unzip..."
    sudo apt-get update
    sudo apt-get install -y unzip
fi

# Unzip the downloaded file
unzip awscliv2.zip

# Install AWS CLI
sudo ./aws/install

# Test installation
/usr/local/bin/aws --version
aws --version

# Optional: Remove the zip file
rm /workspace/awscliv2.zip

echo "AWS CLI installation and configuration complete."

echo "Setting up AWS credentials ..."

# Get AWS credentials 
get_input() {
    read -p "$1: " value
    echo $value
}
aws_access_key_id=$(get_input "Enter your AWS access key id")
aws_secret_access_key=$(get_input "Enter your AWS secret access key")

# Create AWS credentials file
mkdir -p ~/.aws
cat <<EOL > ~/.aws/credentials
[default]
aws_access_key_id = $aws_access_key_id
aws_secret_access_key = $aws_secret_access_key
EOL

echo "AWS credentials setup completes."
