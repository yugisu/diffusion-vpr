### romulan-voyager-4

https://wandb.ai/dmytrii-puzyr-ukrainian-catholic-university/diffusion-vpr/runs/txp1igv5?nw=nwuserdmytriipuzyr

Two UAV-like views.
Overfitting on VisLoc flight 03.
Recall@1 from 0% to 1.6%.
TSNE looks "better".

- augmentations are "in-place" and don't vary within the limits of a single chunk
- UAV-sim augmentations aren't bridging the gap
- need to compare what the backbone "sees" emprically (in a notebook). the backbone can produce poor features for UAV images.
- what if we treat UAV images as a separate modality and use a separate network for UAV vs DiffusionSat for satellite imagery?
- what if we use SD2.1 instead of DiffusionSat as it's not overfit for satellite imagery?
