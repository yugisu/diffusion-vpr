#!/bin/bash
set -e

# ============================================================
# Private credentials — DO NOT commit this file
# ============================================================

GIT_USER_NAME=""
GIT_USER_EMAIL=""
WANDB_API_KEY=""

# ============================================================
# System dependencies
# ============================================================

apt-get update && apt-get install -y unzip gh
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

# ============================================================
# Git & GitHub auth
# ============================================================

git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

gh auth login --web
gh auth setup-git

# ============================================================
# Repositories
# ============================================================

cd /workspace

git clone https://github.com/yugisu/SatDiFuser.git
cd SatDiFuser && git checkout research && cd ..

git clone https://github.com/yugisu/diffusion-vpr.git
cd diffusion-vpr
uv sync

cat > .env <<EOF
VISLOC_ROOT="/workspace/data/visloc"
DIFFUSIONSAT_256_CHCKPT="/workspace/checkpoints/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed"
WANDB_API_KEY="$WANDB_API_KEY"
EOF

# ============================================================
# Data
# ============================================================

mkdir -p /workspace/data && cd /workspace/data

# --- VisLoc full dataset ---
uvx gdown 1xYODANyilEMM3CfWh85APwkTHQeLTcCT -O visloc.zip
unzip visloc.zip -d visloc
mv visloc/'satellite_ coordinates_range.csv' visloc/satellite_coordinates_range.csv
rm -f visloc.zip

# # --- VisLoc example dataset ---
# uvx gdown 16tY7tPZiNIoyAhknvyXnp0jAfccIcHtL -O visloc_example.zip
# unzip visloc_example.zip -d visloc_example
# mv visloc_example/'satellite_ coordinates_range.csv' visloc_example/satellite_coordinates_range.csv
# rm -f visloc_example.zip

# # --- ViLD dataset ---
# wget https://zenodo.org/records/19223815/files/ViLD_dataset.zip?download=1 -O ViLD_dataset.zip
# unzip -q ViLD_dataset.zip -d ViLD_dataset
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
uvx gdown --folder 1VG4yV_fD9UhOa30JzsNRdTwG4cdeJlmX -O checkpoints/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed

echo ""
echo "=== Setup complete ==="
