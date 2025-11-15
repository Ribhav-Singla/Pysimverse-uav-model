#!/bin/bash
set -e

# Install Python dependencies
pip install -r requirements.txt

# Clean install Node dependencies
rm -rf node_modules package-lock.json
npm install

# Install PM2 globally
npm install -g pm2
