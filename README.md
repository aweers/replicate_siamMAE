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
utils/video_downloader.sh utils/kinetics_400_train_files.txt 30 data/class1/

chmod +x utils/video_file_validation.sh
cd data/class1/
../../utils/video_file_validation.sh -y
```

This downloads a uniformly sampled subset of roughly 30,000 files and removes all files which are either corrupted, too short or have a different frame rate than 30 fps (removes roughly 25% of files). 
Note that we the method doesn't require use any class labels and therefore we store only the video files in a single folder.

### Data loading
We use [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo/) for the data loading. To recreate the random frame sampling with a gap between 4 and 48 frames we create our own sampler. The sampler can be found in `utils/random_temporal_subsample.py`.
In [this notebook](notebooks/dataloading.ipynb) we show empirically that the sampler works as expected.