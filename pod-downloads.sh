# Data
mkdir -p data
cd /workspace/data

# VisLoc example ds
uvx gdown 16tY7tPZiNIoyAhknvyXnp0jAfccIcHtL -O visloc_example.zip
unzip visloc_example.zip -d visloc_example
mv visloc_example/'satellite_ coordinates_range.csv' visloc_example/satellite_coordinates_range.csv
rm -rf visloc_example.zip

# VisLoc full ds
uvx gdown 1xYODANyilEMM3CfWh85APwkTHQeLTcCT -O visloc.zip
unzip visloc.zip -d visloc
mv visloc/'satellite_ coordinates_range.csv' visloc/satellite_coordinates_range.csv
rm -rf visloc.zip

# Checkpoints
mkdir -p checkpoints
cd /workspace/checkpoints

# Trimmed DiffusionSat 256 checkpoint at 150k steps.
uvx gdown --folder 1VG4yV_fD9UhOa30JzsNRdTwG4cdeJlmX -O checkpoints/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed