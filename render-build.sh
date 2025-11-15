#!/bin/bash
set -e

# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
npm install

# Install PM2 globally
npm install -g pm2
