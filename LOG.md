### romulan-voyager-4

https://wandb.ai/dmytrii-puzyr-ukrainian-catholic-university/diffusion-vpr/runs/txp1igv5

Two UAV-like views.
Overfitting on VisLoc flight 03.
Recall@1 from 0% to 1.6%.
TSNE looks "better".

- augmentations are "in-place" and don't vary within the limits of a single chunk
- UAV-sim augmentations aren't bridging the gap
- need to compare what the backbone "sees" emprically (in a notebook). the backbone can produce poor features for UAV images.
- what if we treat UAV images as a separate modality and use a separate network for UAV vs DiffusionSat for satellite imagery?
- what if we use SD2.1 instead of DiffusionSat as it's not overfit for satellite imagery?

### trying to optimize inference time

[notebookes/2-evaluate-timesteps.py](notebookes/2-evaluate-timesteps.py), results: [results/2-evaluate-timesteps.csv](notebookes/2-evaluate-timesteps.py).

Findings: after testing on a single flight with UAV/satellite data, we can cut num_timesteps from 50 to 10, and choose save_timesteps as [8, 7] instead of [48, 46, 42] with a better R@1 metric.

### hirogen-blood-wine-6

https://wandb.ai/dmytrii-puzyr-ukrainian-catholic-university/diffusion-vpr/runs/fhejf44a

Supervised training of the embedder head. 10 epochs, R@1 19%, R@5 49%, R@10 67%.

```
FLIGHT_IDS = ["01", "02", "03", "05", "09", "10"]
VAL_FLIGHT_ID = "03"
```

It seems like in the end the model learns something. We need more epochs!

- Claude tells me there's a possible issue with re-ranking given the gap between R@K metrics
- also, we can do better with hard negative mining / multi-positive chunks
- more epochs, second training phase

### bajoran-queen-7

https://wandb.ai/dmytrii-puzyr-ukrainian-catholic-university/diffusion-vpr/runs/c0omo2f0

Supervised training of the embedder head. 50 epochs, R@1 9.6%, R@5 27%, R@10 38%.

```
FLIGHT_IDS = ["01", "02", "04", "05", "06", "08", "09", "10", "11"]
VAL_FLIGHT_ID = "03"
```

In contrary to the previous experiment, the training data didn't contain the validation flight, "degrading" performance by a large margin. Basically, this experiment shows that the model doesn't seem to be generalizing well with the current supervised training setup.

This sets a *lightweight supervised adaptation* of our system and is a weak baseline that improves upon zero-shot 2% R@1 by a factor of 5.

There are ways of improving this:
- inspect other objectives that will force the model cross the domain gap between UAV and satellite imagery (InfoNCE might not be a suitable fit?)
- inspect augmentation logic
- contrastive loss with hard negative mining ?

**Actual stuff to improve:**
- training data is scuffed, augmentations introduce black corners making images borderline unusable; also, strong overly zoom ins, need to tune that
- no negatives included in the training data as the model has only been trained on PAIRS, not on negative satellite imagery for which there is no corresponding UAV shots
- ~~the embeddings should NOT be normalized during training~~ InfoNCE requires normalized embeddings due to dot-product.
