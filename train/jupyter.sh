#!/bin/bash
set -e

# Variables (Customize these as needed)
JUPYTER_PASSWORD="your_secure_password_here"
JUPYTER_PORT=8888
USER_NAME="ubuntu"
JUPYTER_HOME="/home/$USER_NAME"

source activate pytorch
pip install jupyter

# Set Jupyter password and hash
JUPYTER_HASH=$(python3 -c "
from notebook.auth import passwd
print(passwd('$JUPYTER_PASSWORD'))
")
ESCAPED_HASH=$(echo "$JUPYTER_HASH" | sed 's/\$/\\$/g')

# Create directories for SSL certificates
# SSL_DIR="$JUPYTER_HOME/.jupyter/ssl"
# sudo -u $USER_NAME mkdir -p $SSL_DIR

# Generate self-signed SSL certificate
# sudo -u $USER_NAME bash -c "
# openssl req -x509 -nodes -days 365 \
#     -subj '/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=localhost' \
#     -newkey rsa:2048 \
#     -keyout $SSL_DIR/jupyter.key \
#     -out $SSL_DIR/jupyter.crt
# "

# Set permissions for SSL files
chown -R $USER_NAME:$USER_NAME $JUPYTER_HOME/.jupyter
# sudo chmod 600 $SSL_DIR/jupyter.key
# sudo chmod 644 $SSL_DIR/jupyter.crt

# Configure Jupyter Notebook with SSL
sudo -u $USER_NAME -H bash -c "cat > $JUPYTER_HOME/.jupyter/jupyter_notebook_config.py << EOF
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = $JUPYTER_PORT
c.NotebookApp.password = '$ESCAPED_HASH'
c.NotebookApp.allow_origin = '*'
c.NotebookApp.allow_remote_access = True
c.NotebookApp.open_browser = False
c.NotebookApp.notebook_dir = '$JUPYTER_HOME'
# SSL Configuration
# c.NotebookApp.certfile = u'$SSL_DIR/jupyter.crt'
# c.NotebookApp.keyfile = u'$SSL_DIR/jupyter.key'
EOF
"

# Fix permissions
chown -R $USER_NAME:$USER_NAME $JUPYTER_HOME/.jupyter
# Ensure log files exist and set permissions
touch /var/log/jupyter.log /var/log/jupyter_err.log
chown $USER_NAME:$USER_NAME /var/log/jupyter.log /var/log/jupyter_err.log
chmod 644 /var/log/jupyter.log /var/log/jupyter_err.log

# Find Jupyter executable path
JUPYTER_PATH=$(which jupyter)

# Create systemd service file for Jupyter
bash -c "cat > /etc/systemd/system/jupyter.service << EOF
[Unit]
Description=Jupyter Notebook Server
After=network.target

[Service]
Type=simple
User=$USER_NAME
ExecStart=/bin/bash -c 'source activate pytorch && jupyter notebook'
WorkingDirectory=$JUPYTER_HOME
Restart=always
RestartSec=10
Environment=PATH=/usr/bin:/usr/local/bin
Environment=HF_HUB_CACHE=$HF_HUB_CACHE

# Logging
StandardOutput=append:/var/log/jupyter.log
StandardError=append:/var/log/jupyter_err.log

[Install]
WantedBy=multi-user.target
EOF
"

# Reload systemd daemon, enable and start Jupyter service
systemctl daemon-reload
systemctl enable jupyter
systemctl start jupyter
