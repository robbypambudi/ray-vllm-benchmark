#!/bin/bash

# Docker & GPU Installation Script
# Kompatibel dengan Ubuntu/Debian
# Author: Generated Script
# Date: $(date)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "Script tidak boleh dijalankan sebagai root!"
        print_status "Gunakan: ./install.sh"
        exit 1
    fi
}

# Function to detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$NAME
        VER=$VERSION_ID
    else
        print_error "Tidak dapat mendeteksi OS"
        exit 1
    fi

    print_status "Detected OS: $OS $VER"
}

# Function to update system
update_system() {
    print_status "Updating system packages..."
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y curl wget gnupg lsb-release apt-transport-https ca-certificates software-properties-common
}

# Function to install Docker
install_docker() {
    print_status "Installing Docker..."

    # Remove old versions
    sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt update
    sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    # Add user to docker group
    sudo usermod -aG docker $USER

    # Enable and start Docker service
    sudo systemctl enable docker
    sudo systemctl start docker

    print_success "Docker installed successfully!"
}

# Function to check GPU
check_gpu() {
    print_status "Checking for GPU..."

    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_success "NVIDIA GPU detected: $GPU_INFO"
        return 0
    elif lspci | grep -i amd | grep -i vga &> /dev/null; then
        print_warning "AMD GPU detected. Script akan install ROCm support."
        return 1
    else
        print_warning "No dedicated GPU detected. Skipping GPU setup."
        return 2
    fi
}

# Function to install NVIDIA drivers
install_nvidia_drivers() {
    print_status "Installing NVIDIA drivers..."

    # Add NVIDIA PPA
    sudo add-apt-repository ppa:graphics-drivers/ppa -y
    sudo apt update

    # Install recommended driver
    sudo ubuntu-drivers autoinstall

    print_warning "NVIDIA drivers installed. Reboot required!"
}

# Function to install NVIDIA Container Toolkit
install_nvidia_docker() {
    print_status "Installing NVIDIA Container Toolkit..."

    # Configure the production repository (official method from NVIDIA docs)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # Update the packages list from the repository
    sudo apt-get update

    # Install the NVIDIA Container Toolkit packages
    sudo apt-get install -y nvidia-container-toolkit

    print_success "NVIDIA Container Toolkit installed!"
}

# Function to configure Docker for NVIDIA Container Toolkit
configure_docker_nvidia() {
    print_status "Configuring Docker for NVIDIA Container Toolkit..."

    # Configure the container runtime using nvidia-ctk command
    sudo nvidia-ctk runtime configure --runtime=docker

    # Restart the Docker daemon
    sudo systemctl restart docker

    print_success "Docker configured for NVIDIA Container Toolkit!"
}

# Function to install AMD ROCm (for AMD GPUs)
install_rocm() {
    print_status "Installing AMD ROCm..."

    # Add ROCm repository
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

    sudo apt update
    sudo apt install -y rocm-dkms rocm-libs rocm-dev

    # Add user to render group
    sudo usermod -aG render $USER

    print_success "AMD ROCm installed!"
}

# Function to test Docker installation
test_docker() {
    print_status "Testing Docker installation..."

    if sudo docker run hello-world &> /dev/null; then
        print_success "Docker is working correctly!"
    else
        print_error "Docker test failed!"
        return 1
    fi
}

# Function to test GPU in Docker
test_gpu_docker() {
    print_status "Testing GPU in Docker..."

    if command -v nvidia-smi &> /dev/null; then
        print_status "Running GPU test with official NVIDIA base image..."
        if sudo docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            print_success "NVIDIA GPU is working in Docker!"

            # Show GPU info
            print_status "GPU Information:"
            sudo docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi -L
        else
            print_error "NVIDIA GPU test in Docker failed!"
            print_warning "Try running: docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi"
        fi
    else
        print_warning "Skipping GPU test (no NVIDIA GPU detected)"
    fi
}

# Function to install Docker Compose (standalone)
install_docker_compose() {
    print_status "Installing Docker Compose..."

    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep -oP '"tag_name": "\K(.*)(?=")')
    sudo curl -L "https://github.com/docker/compose/releases/download/$COMPOSE_VERSION/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose

    print_success "Docker Compose $COMPOSE_VERSION installed!"
}

# Function to show post-installation info
show_post_install_info() {
    print_success "Installation completed!"
    echo ""
    print_status "Post-installation steps:"
    echo "1. Logout and login again (or reboot) to apply group changes"
    echo "2. Test Docker: docker run hello-world"

    if command -v nvidia-smi &> /dev/null; then
        echo "3. Test GPU: docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi"
    fi

    echo ""
    print_status "Useful commands:"
    echo "- docker --version"
    echo "- docker-compose --version"
    echo "- nvidia-smi (for NVIDIA GPU)"
    echo "- rocm-smi (for AMD GPU)"
}

# Main execution
main() {
    clear
    echo "================================================"
    echo "    Docker & GPU Installation Script"
    echo "================================================"
    echo ""

    check_root
    detect_os

    # Update system
    update_system

    # Install Docker
    install_docker

    # Install Docker Compose
    install_docker_compose

    # Check and install GPU support
    check_gpu
    gpu_status=$?

    if [[ $gpu_status -eq 0 ]]; then
        # NVIDIA GPU detected
        if ! command -v nvidia-smi &> /dev/null; then
            print_status "NVIDIA drivers not found. Installing..."
            install_nvidia_drivers
        fi
        install_nvidia_docker
        configure_docker_nvidia
    elif [[ $gpu_status -eq 1 ]]; then
        # AMD GPU detected
        install_rocm
    fi

    # Test installations
    test_docker

    if [[ $gpu_status -eq 0 ]]; then
        test_gpu_docker
    fi

    # Show final information
    show_post_install_info

    print_warning "Please reboot your system to complete the installation!"
}

# Run main function
main "$@"