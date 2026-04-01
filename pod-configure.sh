# Install deps
curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get update
apt-get install -y unzip gh
source $HOME/.local/bin/env

# Setup repos
git clone https://github.com/yugisu/SatDiFuser.git
git clone https://github.com/yugisu/diffusion-vpr.git

cd ~/SatDiFuser
git checkout research

cd ~/diffusion-vpr
uv sync
cp .env.example .env

# Auth
gh auth login --web
gh auth setup-git