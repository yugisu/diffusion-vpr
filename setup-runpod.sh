#!/bin/bash
set -e

### IMPORTANT: Machine requirements
# - 30GB of container disk
# - 90GB of pod volume
# - A100 instance is good

### Copy this file to the runpod instance.
# scp -P PORT -i ~/.ssh/id_ed25519-personal setup-runpod.sh root@RUNPOD_IP:~/setup-runpod.sh

### To start a persistent tmux session for training:
# tmux new -s training
### To reconnect to the tmux session after disconnecting:
# tmux attach -t training

# ============================================================
# NOTE: Required private env variables. Should be provided by the instance.
# ============================================================

# GIT_USER_NAME=""
# GIT_USER_EMAIL=""
# GH_TOKEN=""
# WANDB_API_KEY=""
# HF_TOKEN=""

for var in GIT_USER_NAME GIT_USER_EMAIL GH_TOKEN WANDB_API_KEY HF_TOKEN; do
  [ -n "${!var}" ] || { echo "ERROR: $var is not set"; exit 1; }
done

# ============================================================
# System dependencies
# ============================================================

apt-get update && apt-get install -y unzip gh tmux
command -v uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
command -v claude &>/dev/null || curl -fsSL https://claude.ai/install.sh | bash
source "$HOME/.local/bin/env"

# ============================================================
# Git & GitHub auth
# ============================================================

git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

echo "$GH_TOKEN" | gh auth login --with-token
gh auth setup-git

# ============================================================
# Repositories
# ============================================================

cd /root

[ -d SatDiFuser ] || git clone https://github.com/yugisu/SatDiFuser.git
cd SatDiFuser && git checkout research && cd ..
[ -d dift ] || git clone https://github.com/yugisu/dift.git
cd dift && git checkout research && cd ..

[ -d diffusion-vpr ] || git clone https://github.com/yugisu/diffusion-vpr.git
cd diffusion-vpr
uv sync

# Populate .env file
cat > .env <<EOF
VISLOC_ROOT="/workspace/data/visloc"
SECO_ROOT="/workspace/data/seco_100k/seasonal_contrast_100k"
DIFFUSIONSAT_256_CHCKPT="/workspace/checkpoints/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed"
HF_HOME="/workspace/.hugging_face"
WANDB_API_KEY="$WANDB_API_KEY"
HF_TOKEN="$HF_TOKEN"
EOF

STATE_DIR="/workspace/state"

# Preserve state
mkdir -p $STATE_DIR/lightning_logs/
[ -e lightning_logs ] || ln -s $STATE_DIR/lightning_logs/ lightning_logs
mkdir -p $STATE_DIR/wandb/
[ -e wandb ] || ln -s $STATE_DIR/wandb/ wandb
mkdir -p $STATE_DIR/checkpoints/
[ -e checkpoints ] || ln -s $STATE_DIR/checkpoints/ checkpoints

# ============================================================
# Data
# ============================================================

mkdir -p /workspace/data && cd /workspace/data

# --- VisLoc full dataset ---
if [ ! -d /workspace/data/visloc ]; then
  uvx gdown 16vbbiV93rdQL2v_66ccrxICtROugkw2c -O visloc.zip
  unzip -q -u visloc.zip -d visloc
  mv visloc/'satellite_ coordinates_range.csv' visloc/satellite_coordinates_range.csv
  rm -f visloc.zip
fi

# # --- VisLoc example dataset ---
# uvx gdown 16tY7tPZiNIoyAhknvyXnp0jAfccIcHtL -O visloc_example.zip
# unzip -q -u visloc_example.zip -d visloc_example
# mv visloc_example/'satellite_ coordinates_range.csv' visloc_example/satellite_coordinates_range.csv
# rm -f visloc_example.zip

# # --- SeCo dataset ---
# uvx gdown 1pEcd78S5t_Bk76dNXRCMZuqqwI_ecpfc -O seco_100k.zip
# unzip -q -u seco_100k.zip -d seco_100k
# rm -f seco_100k.zip

# # --- ViLD dataset ---
# wget https://zenodo.org/records/19223815/files/ViLD_dataset.zip?download=1 -O ViLD_dataset.zip
# unzip -q -u ViLD_dataset.zip -d ViLD_dataset
# rm -f ViLD_dataset.zip

# # --- SSL4EO-S12 example dataset ---
# uvx gdown 1sRWcYbaWs-efXza6kw03GlJQdZHq5iRN -O SSL4EO-S12_example.tar.gz
# mkdir -p SSL4EO-S12_example
# tar -xzf SSL4EO-S12_example.tar.gz -C ./SSL4EO-S12_example/
# rm -f SSL4EO-S12_example.tar.gz

# ============================================================
# Checkpoints
# ============================================================

mkdir -p /workspace/checkpoints && cd /workspace/checkpoints

# Trimmed DiffusionSat 256 checkpoint at 150k steps
[ -d /workspace/checkpoints/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed ] || uvx gdown --folder 1VG4yV_fD9UhOa30JzsNRdTwG4cdeJlmX -O finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed

# ============================================================
# VS Code CLI + extensions
# ============================================================

# curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o /tmp/vscode_cli.tar.gz
# tar -xf /tmp/vscode_cli.tar.gz -C /usr/local/bin/
# rm /tmp/vscode_cli.tar.gz

# code --install-extension "astral-sh.type"
# code --install-extension "ms-python.pythonpe"
# code --install-extension "ms-toolsai.jupyterpe"
# code --install-extension "charliermarsh.ruffpe"
# code --install-extension "anthropic.claude-codepe"


echo ""
echo "=== Setup complete ==="
