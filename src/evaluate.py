import torch
import torch.nn as nn
import numpy as np
from pytorchvideo.data import LabeledVideoDataset, UniformClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    Div255,
    Permute
)

from torchvision.transforms import (
    Compose,
    CenterCrop,
    Resize
)
import torchmetrics
from tqdm import tqdm

import os
import sys
sys.path.insert(1, '../src')
from vit import Embedding, ViT_Encoder, Masking
from wandb_helper import download_model_from_run

def unnormalize(tensor):
    """
    Unnormalizes a tensor by applying the reverse transformation of the normalization process.

    Args:
        tensor (torch.Tensor): The tensor to be unnormalized.

    Returns:
        torch.Tensor: The unnormalized tensor.

    """
    unnormalized = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return (torch.clamp(unnormalized, 0, 1) * 255).to(dtype=torch.uint8)

def create_label_list(directory_path, label):
    """
    Create a list of labeled directories (necessary for loading videos as directories
    of frames).

    Args:
        directory_path (str): The path to the directory containing the directories to be labeled.
        label (str): The label to assign to the directories.

    Returns:
        list: A list of tuples, where each tuple contains the path of a labeled directory and its corresponding label.
    """
    items = os.listdir(directory_path)
    dirs = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    labeled_dirs = [(os.path.join(directory_path, dir), label) for dir in dirs]

    return labeled_dirs

def get_features(frame, upsample=True, pos_embed_factor=1.0):
    """
    Extracts features from a given frame.

    Args:
        frame (torch.Tensor): The input frame.
        upsample (bool, optional): Whether to upsample the features. Defaults to True.
        pos_embed_factor (float, optional): The factor to scale the positional embeddings. Defaults to 1.0.

    Returns:
        torch.Tensor: The extracted features.
    """
    if frame.shape[0] == 3:
        features = frame.unsqueeze(0)
    else:
        features = frame
    
    with torch.no_grad():
        features = embedding.embedding(features).view(features.shape[0], -1, 768)
        features -= embedding.pos_embed[:, :features.shape[1], :] * (1-pos_embed_factor)
        features = encoder(features)

    n = features.shape[1]
    img_size = int(np.sqrt(n))

    features = features.reshape(features.shape[0], img_size, img_size, -1).permute(0, 3, 1, 2)
    # scale to 224x224
    if upsample:
        features = nn.functional.interpolate(features, size=(frame.shape[2], frame.shape[3]), mode='nearest')
    return features

def create_mask(size, n):
    """
    Create a binary mask of size `size` where each pixel within a distance `n` from any other pixel is set to 1,
    and all other pixels are set to 0.

    Args:
        size (int): The size of the mask (width and height).
        n (float): The maximum distance between pixels for them to be considered neighbors.

    Returns:
        torch.Tensor: The binary mask of size `size`.
    """
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
    coords = torch.stack((x, y), dim=-1).reshape(-1, 2).float()

    dists = torch.cdist(coords, coords)
    mask = (dists <= n).float()

    return mask

def propagate_labels_multi_frames(features_previous, features_current, labels_previous, k, radius=5):
    """
    Propagates labels from previous frames to the current frame based on feature affinities.

    Args:
        features_previous (torch.Tensor): Feature maps of the previous frames. Shape: (n, 196, 768)
        features_current (torch.Tensor): Feature maps of the current frame. Shape: (196, 768)
        labels_previous (torch.Tensor): Labels of the previous frames. Shape: (n, 196, 3)
        k (int): Number of top affinities to consider for label propagation.
        radius (int, optional): Radius for creating the affinity mask. Defaults to 5.

    Returns:
        torch.Tensor: Propagated labels for the current frame. Shape: (196, 3)
    """
    T = 1 # temperature parameter

    # Reshape the feature maps
    features_previous = features_previous.reshape(features_previous.shape[0], features_previous.shape[1], -1).transpose(1, 2) # n x 196 x 768
    features_current = features_current.reshape(features_current.shape[0], -1).transpose(0, 1)                                # 196 x 768
    labels_previous = labels_previous.reshape(labels_previous.shape[0], labels_previous.shape[1], -1).transpose(1, 2)         # n x 196 x 3

    affinities = []
    for frame_nr in range(features_previous.shape[0]):
        # Compute the affinity between the features in the previous and current frames
        affinity = torch.matmul(features_previous[frame_nr, :, :], features_current.T) # 196 x 196
        affinity = torch.nn.functional.softmax(affinity/T, dim=1)
        mask = create_mask(int(affinity.shape[0]**0.5), radius)

        affinity = affinity * mask

        affinities.append(affinity)
    
    affinities = torch.stack(affinities, dim=0) # shape: n x 196 x 196

    labels_next = torch.zeros((labels_previous.shape[1], labels_previous.shape[2]))
    for i in range(labels_previous.shape[1]): # loop over all pixels
        averaged_value = torch.zeros((labels_previous.shape[2]))
        total_weight = 0
        for j in range(labels_previous.shape[0]):
            value, index = torch.sort(affinities[j, :, i])
            
            value = value[-k:]
            index = index[-k:]

            averaged_value += torch.matmul(value, labels_previous[j, index, :])
            total_weight += torch.sum(value)
        averaged_value /= total_weight
        labels_next[i, :] = averaged_value

    return labels_next

def evaluate_video(video, annotation, pos_embed_factor=1.0, queue_length=20, k=7, neighbor=1, interpolation_mode='bilinear'):
    """
    Evaluate the video by calculating the Jaccard index and F1 score for each frame. Labels are propagated from
    the previous frames to the current frame.

    Args:
        video (torch.Tensor): The video frames.
        annotation (torch.Tensor): The ground truth annotation for each frame.
        pos_embed_factor (float, optional): The position embedding factor. Defaults to 1.0.
        queue_length (int, optional): The length of the queue frames. Defaults to 20.
        k (int, optional): The number of nearest neighbors to consider. Defaults to 7.
        neighbor (int, optional): The number of neighbors to propagate labels from. Defaults to 1.
        interpolation_mode (str, optional): The interpolation mode for resizing frames. Defaults to 'bilinear'.

    Returns:
        float: The average Jaccard index for all frames.
        float: The average F1 score for all frames.
    """
    video_length = video.shape[0]
    
    # Calculate features for all frames
    features = get_features(video, upsample=False, pos_embed_factor=pos_embed_factor)
    
    jaccard = torchmetrics.classification.MultilabelJaccardIndex(3, average='micro', validate_args=False)
    f1 = torchmetrics.classification.MultilabelF1Score(3, average='micro')
    for i in range(video_length-queue_length):
        # Prepare queue frames
        features_previous = features[i:i+queue_length]
        labels_previous = nn.functional.interpolate(torch.stack([annotation[i+j] for j in range(queue_length)]).unsqueeze(0), size=(3, 14, 14), mode='nearest').squeeze(0)

        # Prepare next frame
        features_next = features[i+queue_length]

        # Propagate labels
        labels_next = propagate_labels_multi_frames(features_previous, features_next, labels_previous, k, neighbor).reshape(14, 14, 3).permute(2, 0, 1)

        # Calculate jaccard index
        next_labels = nn.functional.interpolate(unnormalize(labels_next).unsqueeze(0), size=(224, 224), mode=interpolation_mode).squeeze(0).permute(1, 2, 0)

        ground_truth = unnormalize(annotation[i+queue_length]).permute(1, 2, 0)
        ground_truth = ((ground_truth / ground_truth.max()) > 0.5).to(dtype=torch.uint8)
        
        prediction = ((next_labels/next_labels.max()) > 0.5).to(dtype=torch.uint8)
        # Choose the class with the highest probability (one-hot encoding)
        #prediction = torch.zeros_like(next_labels)
        #prediction[torch.arange(next_labels.shape[0]), torch.arange(next_labels.shape[1]), next_labels.argmax(dim=2)] = 1

        val = jaccard(prediction.permute(2, 0, 1).unsqueeze(0), ground_truth.permute(2, 0, 1).unsqueeze(0))
        f1(prediction.permute(2, 0, 1).unsqueeze(0), ground_truth.permute(2, 0, 1).unsqueeze(0))
        #plt.subplot(1, 2, 1)
        #plt.imshow(prediction*255)
        #plt.subplot(1, 2, 2)
        #plt.imshow(ground_truth*255)
        #plt.title(f"Jaccard Index: {val.item():.4f}")
        #plt.show()

    return jaccard.compute().item(), f1.compute().item()

# Settings
device = "cpu"
run_id = "4153hlx9"

artifact, cfg = download_model_from_run("aweers/dd2412-exploration/"+run_id)

artifact_dir = artifact.download()

if "GELU" in cfg['mlp_activation']:
    cfg['mlp_activation'] = nn.GELU()
elif "ReLU" in cfg['mlp_activation']:
    cfg['mlp_activation'] = nn.ReLU()
else:
    raise NotImplementedError("Activation function not implemented")

encoder = ViT_Encoder(cfg['D'], cfg['encoder_heads'], (cfg['encoder_mlp_dim'], cfg['mlp_activation']), cfg['encoder_layers'])
masking = Masking(cfg['D'])
embedding = Embedding(cfg['D'], cfg['patch_size'], cfg['channels'], cfg['image_size']//cfg['patch_size'] * cfg['image_size']//cfg['patch_size'], masking)

encoder.load_state_dict(torch.load(artifact_dir + "/encoder.pt", map_location=torch.device(device)))
embedding.load_state_dict(torch.load(artifact_dir + "/embedding.pt", map_location=torch.device(device)))

label = {"category": "example"}
data_list = create_label_list("DAVIS/JPEGImages/480p/", label)
annotation_list = create_label_list("DAVIS/Annotations/480p/", label)

transform = Compose(
    [
    ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
            Resize(cfg['image_size']),
            CenterCrop(cfg['image_size']),
            Div255(),
            # mean and std from ImageNet
            Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            # Permute to get Frames, Channel, Height, Width
            Permute((1, 0, 2, 3))
            ]
        ),
        ),
    ]
)
clip_duration = 20 # needs to be minimum as long as the longest video in the dataset
data = LabeledVideoDataset(data_list, clip_sampler=UniformClipSampler(clip_duration=clip_duration), decode_audio=False, transform=transform)
annotations = LabeledVideoDataset(annotation_list, clip_sampler=UniformClipSampler(clip_duration=clip_duration), decode_audio=False, transform=transform)

dloader = torch.utils.data.DataLoader(
    data,
    batch_size=1,
    num_workers=0,
    shuffle=False
)
annot_loader = torch.utils.data.DataLoader(
    annotations,
    batch_size=1,
    num_workers=0,
    shuffle=False
)

pbar = tqdm(zip(dloader, annot_loader), total=data.num_videos)

js = []
f1s = []
for batch, annot_batch in pbar:
    video = batch['video']
    annotation = annot_batch['video']

    jaccard, f1 = evaluate_video(video[0], annotation[0], pos_embed_factor=1.0, interpolation_mode='bilinear')
    js.append(jaccard)
    f1s.append(f1)
    
    pbar.set_description(f"Mean Jaccard Index: {np.mean(js):.4f} Mean F1 Score: {np.mean(f1s):.4f}")

print(f"Evaluation of run {run_id} finished. Mean Jaccard Index: {np.mean(js):.6f}\tStd: {np.std(js):.6f}\tMean F1 Score: {np.mean(f1s):.6f}\tStd: {np.std(f1s):.6f}")