#!/bin/bash
# setup.sh — Automated CogHealth setup for Raspberry Pi 4
# Run once as: bash setup.sh
# Tested on Raspberry Pi OS Bookworm (64-bit)

set -euo pipefail

COGHEALTH_DIR="$HOME/coghealth"
VENV_DIR="$COGHEALTH_DIR/venv"
SERVICE_NAME="coghealth"
LOG_FILE="$COGHEALTH_DIR/logs/setup.log"

# ── Colors ────────────────────────────────────────────────────────────────────
GRN='\033[0;32m'; YLW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GRN}[OK]${NC}  $1"; }
warn() { echo -e "${YLW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
step() { echo -e "\n${YLW}>>> $1${NC}"; }

mkdir -p "$COGHEALTH_DIR/logs"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "  CogHealth Edge System — Setup Script"
echo "  $(date)"
echo "============================================"

# ── 1. System packages ────────────────────────────────────────────────────────
step "Installing system packages"
sudo apt-get update -qq
sudo apt-get install -y \
    python3.9 python3.9-venv python3-pip \
    pigpio \
    python3-rpi.gpio \
    libhdf5-dev \
    libatlas-base-dev \
    libjpeg-dev \
    git curl \
    2>/dev/null
ok "System packages installed"

# ── 2. Enable and start pigpio daemon ─────────────────────────────────────────
step "Configuring pigpio"
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
sleep 1
if sudo systemctl is-active --quiet pigpiod; then
    ok "pigpiod running"
else
    warn "pigpiod not running — LED control will use mock mode"
fi

# ── 3. Python virtual environment ────────────────────────────────────────────
step "Creating Python virtual environment"
python3.9 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip wheel
ok "Virtual environment ready: $VENV_DIR"

# ── 4. Python dependencies ────────────────────────────────────────────────────
step "Installing Python dependencies"
pip install --quiet -r "$COGHEALTH_DIR/requirements.txt"
# TFLite runtime for ARM
pip install --quiet tflite-runtime 2>/dev/null || \
    warn "tflite-runtime not available — using full TensorFlow"
ok "Python dependencies installed"

# ── 5. Directory structure ────────────────────────────────────────────────────
step "Creating data directories"
mkdir -p \
    "$COGHEALTH_DIR/data/raw" \
    "$COGHEALTH_DIR/data/features" \
    "$COGHEALTH_DIR/models" \
    "$COGHEALTH_DIR/logs" \
    "$COGHEALTH_DIR/web/static" \
    "$COGHEALTH_DIR/web/templates"
ok "Directories created"

# ── 6. GPIO permissions ───────────────────────────────────────────────────────
step "Configuring GPIO permissions"
if ! groups "$USER" | grep -q gpio; then
    sudo usermod -aG gpio,input "$USER"
    warn "Added $USER to gpio/input groups — reboot required to take effect"
fi

# ── 7. Autostart on login (X11 — for keyboard monitoring) ────────────────────
step "Configuring display environment for pynput"
XAUTHOR_FILE="$HOME/.Xauthority"
ENV_FILE="$COGHEALTH_DIR/.env"
cat > "$ENV_FILE" <<EOF
DISPLAY=:0
XAUTHORITY=$XAUTHOR_FILE
HOME=$HOME
PYTHONPATH=$COGHEALTH_DIR
EOF
ok "Environment file: $ENV_FILE"

# ── 8. Systemd service ────────────────────────────────────────────────────────
step "Installing systemd service"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

sudo tee "$SERVICE_FILE" > /dev/null <<EOF
[Unit]
Description=CogHealth Behavioral Monitoring System
After=network.target graphical-session.target pigpiod.service
Wants=pigpiod.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$COGHEALTH_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$VENV_DIR/bin/python $COGHEALTH_DIR/orchestrator.py
Restart=always
RestartSec=10
StandardOutput=append:$COGHEALTH_DIR/logs/coghealth.log
StandardError=append:$COGHEALTH_DIR/logs/coghealth.log
KillMode=control-group
TimeoutStopSec=30

[Install]
WantedBy=graphical-session.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
ok "Service installed: $SERVICE_FILE"

# ── 9. Initial model pre-training ────────────────────────────────────────────
step "Running initial model pre-training (≈5 min on Pi 4)"
echo "This trains the LSTM autoencoder on synthetic baseline data."
echo "You can skip this and let the system train on first launch."
read -t 30 -p "Run pre-training now? [y/N] " PRETRAIN || true
if [[ "${PRETRAIN:-n}" =~ ^[Yy]$ ]]; then
    source "$VENV_DIR/bin/activate"
    cd "$COGHEALTH_DIR"
    python model.py data models
    ok "Model pre-training complete"
else
    warn "Skipping pre-training — will run on first system start"
fi

# ── 10. Hardware test ─────────────────────────────────────────────────────────
step "LED hardware test"
read -t 15 -p "Run LED test? Requires GPIO wiring [y/N] " LED_TEST || true
if [[ "${LED_TEST:-n}" =~ ^[Yy]$ ]]; then
    source "$VENV_DIR/bin/activate"
    python - <<'PYEOF'
import time
from led import LEDController
ctrl = LEDController()
for level in ["baseline", "low", "moderate", "elevated", "high", "offline"]:
    print(f"  LED → {level}")
    ctrl.set_level(level, duration=2.0)
    time.sleep(3)
ctrl.stop()
print("LED test complete")
PYEOF
fi

# ── 11. Firewall (optional, expose web UI on LAN) ────────────────────────────
step "Optional: open port 5000 for web dashboard"
read -t 15 -p "Allow port 5000 through firewall? [y/N] " UFW || true
if [[ "${UFW:-n}" =~ ^[Yy]$ ]]; then
    sudo ufw allow 5000/tcp 2>/dev/null || warn "ufw not installed"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo -e "${GRN}  Setup complete!${NC}"
echo "============================================"
echo ""
echo "To start the system now:"
echo "  sudo systemctl start $SERVICE_NAME"
echo ""
echo "To check status:"
echo "  sudo systemctl status $SERVICE_NAME"
echo ""
echo "To view logs:"
echo "  tail -f $COGHEALTH_DIR/logs/coghealth.log"
echo ""
echo "Web dashboard (after system starts):"
IP=$(hostname -I | awk '{print $1}')
echo "  http://$IP:5000"
echo ""
echo "SUS questionnaire:"
echo "  http://$IP:5000/sus"
echo ""
echo -e "${YLW}NOTE: A reboot is recommended before first use.${NC}"
echo ""
