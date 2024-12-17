#!/bin/bash


# Check if curl is installed, if not use wget
if command -v curl > /dev/null; then
  DOWNLOAD_CMD="curl -sSL"
else
  DOWNLOAD_CMD="wget -q --show-progress"
fi

# Check if unzip is installed, if not install it
if ! command -v unzip > /dev/null; then
  echo "unzip not found, installing..."
  sudo apt-get install -y unzip
fi

cd ~
$DOWNLOAD_CMD "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip && rm -f awscliv2.zip

# Install AWS CLI
./aws/install --install-dir ~/aws-cli --bin-dir ~/bin
echo 'export PATH=~/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Test installation
aws --version

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
