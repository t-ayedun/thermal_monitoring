#!/bin/bash
# pi_setup.sh - Run this on the Raspberry Pi

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip i2c-tools git

# Enable I2C
sudo raspi-config nonint do_i2c 0

# Install Python packages
pip3 install adafruit-circuitpython-mlx90640 numpy matplotlib pandas

# Create project directory
mkdir -p /home/$USER/thermal_project
cd /home/$USER/thermal_project

# Set up systemd service for auto-start
sudo tee /etc/systemd/system/thermal-monitor.service > /dev/null <<EOF
[Unit]
Description=Thermal Camera Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/thermal_project
ExecStart=/usr/bin/python3 thermal_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable thermal-monitor.service

echo "Setup complete! Reboot to start monitoring automatically."