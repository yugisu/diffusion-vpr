# Data
mkdir -p data

# VisLoc example ds
uvx gdown 16tY7tPZiNIoyAhknvyXnp0jAfccIcHtL -O visloc_example.zip
unzip visloc_example.zip -d data/visloc_example
rm -rf visloc_example.zip

# Checkpoints
mkdir -p checkpoints

# Trimmed DiffusionSat 256 checkpoint at 150k steps.
uvx gdown --folder 1VG4yV_fD9UhOa30JzsNRdTwG4cdeJlmX -O checkpoints/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed