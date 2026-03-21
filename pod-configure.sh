# Install deps
curl -LsSf https://astral.sh/uv/install.sh | sh
apt-get update
apt-get install -y unzip gh

# Setup repos
git clone https://github.com/yugisu/SatDiFuser.git && cd SatDiFuser && git checkout research && cd ..

git clone https://github.com/yugisu/diffusion-vpr.git && cd diffusion-vpr && uv sync

# Auth
# gh auth login --web
# gh auth setup-git