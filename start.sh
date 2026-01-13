#!/bin/bash
# Start Script for NeuralOps
# © 2025 Neural Operation's & Holding's LLC. All rights reserved.

echo "Starting NeuralOps Server..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed. Please install Node.js 20+ to continue."
    exit 1
fi

# Check Node.js version
NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 20 ]; then
    echo "ERROR: Node.js version 20+ is required. Current version: $(node -v)"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "WARNING: .env file not found. Some features may not work without environment variables."
fi

# Start the Node.js server with memory limit optimized for large datasets (5GB+)
# Use 8GB for Node.js to handle large training data collection
echo "Starting Express server..."
echo "Node.js memory limit: 8GB"
echo "Starting server process..."
NODE_OPTIONS="--max-old-space-size=8192" node runtime/server.js

