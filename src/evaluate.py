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

import matplotlib.pyplot as plt

def unnormalize(tensor):
    unnormalized = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return (torch.clamp(unnormalized, 0, 1) * 255).to(dtype=torch.uint8)

def create_label_list(directory_path, label):
    items = os.listdir(directory_path)
    dirs = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    labeled_dirs = [(os.path.join(directory_path, dir), label) for dir in dirs]

    return labeled_dirs

def get_features(frame, upsample=True, pos_embed_factor=1.0):
    # A forward pass through the encoder part to get the features
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
    # Create a mask to limit the influence of each patch to its neighbors
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size))
    coords = torch.stack((x, y), dim=-1).reshape(-1, 2).float()

    dists = torch.cdist(coords, coords)
    mask = (dists <= n).float()

    return mask

def propagate_labels_multi_frames(features_previous, features_current, labels_previous, k, radius=5):
    T = 1
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

# Performs video label propagation on a video
# The labels of `queue_length` frames are propagated to the next frame
# The labels for each patch are determined by the `k` nearest neighbors (affinity in feature space)
# Each patch's influence is limited to a radius of `neighbor` patches
def evaluate_video(video, annotation, pos_embed_factor=0.01, queue_length=20, k=7, neighbor=1, interpolation_mode='bilinear'):
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

device = "cpu"

run_id = "4153hlx9" # 85 0.95
#run_id = "0qqeq5wm" # 82 0.75
#run_id = "47a3nsuo" # 83 0.75 true_cross
#run_id = "qrk3sza1" # 92 0.5
#run_id = "ckenfh0b"

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
clip_duration = 20
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