# Replicate Siamese Masked Autoencoder
[Siamese Masked Autoencoder](https://arxiv.org/abs/2305.14344)

## Requirements
We provide a requirements.txt file for installing the required packages. 
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## Data
We use a subset of the [Kinetics-400](https://arxiv.org/abs/1705.06950v1) dataset for training and testing. We use the files provided by the [CVD Foundation](https://github.com/cvdfoundation/kinetics-dataset) and some own scripts to download a subset and filter the files as specified in the paper. To recreate our dataset, run the following commands (change the second argument of the `video_downloader.sh` script to the number of files you want to download):
```
chmod +x utils/video_downloader.sh
utils/video_downloader.sh utils/kinetics_400_train_files.txt 30 data

chmod +x utils/video_file_validation.sh
cd data/class1/
../../utils/video_file_validation.sh -y
cd ../../data_val/class1/
../../utils/video_file_validation.sh -y
```

This downloads a uniformly sampled subset of roughly 30,000 files and removes all files which are either corrupted, too short or have a different frame rate than 30 fps (removes roughly 25% of files). It splits the files into a training and validation set with a 90/10 split, which can be adjusted in the script.

Note that the method doesn't require any class labels and therefore we store only the video files and all in a single folder.

The videos could (and should) be preprocessed to shift the video encoding overhead away from each training loop. We use the script `utils/save_frames.sh` to extract the frames from the videos and store them in seperate folders. Note that the script uses paths that are defined in the script itself and therefore need to be adjusted. The script also uses the `ffmpeg` library, which needs to be installed separately.

### Data loading
We use [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo/) for the data loading. To recreate the random frame sampling with a gap between 4 and 48 frames we create our own sampler. We also include the possibility to use a `repeated sampling factor`, such that the overhead of loading and decoding videos is minimized. The sampler can be found in `utils/random_temporal_subsample.py`.
In [this notebook](notebooks/dataloading.ipynb) we show empirically that the sampler works as expected.

### Data Preprocessing (optional, but recommended)
We use the `utils/save_frames.sh` script to extract the frames from the videos and store them in seperate folders. Note that the script uses paths that are defined in the script itself and therefore need to be adjusted. The script also uses the `ffmpeg` library, which needs to be installed separately. Using the preprocessed data can speed up the training process significantly.

## Implementation
We use PyTorch for our implementation. We implemented all required layers and parts of the model in the `src/vit.py` file. Some tests of the ViT implementation can be found in `notebooks/vit-tests.ipynb` and tests of the Siamese Masked Autoencoder can be found in `notebooks/siamMAE.ipynb`.

## Training
We use the `src/train.py` script to train the model. The hyperparameters are specified in the script itself. Those, as well as the whole training process (including model checkpoints and qualitative plots) are synced in a [Weights & Biases](https://wandb.ai/) project.

## Evaluation
We use the `src/evaluate.py` script to evaluate the model. The script loads a model checkpoint and evaluates it on the [DAVIS-2017 dataset](https://davischallenge.org/). 

## Final report
You can find our final report [here](report.pdf). 
