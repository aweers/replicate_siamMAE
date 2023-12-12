import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorchvideo.data import LabeledVideoDataset, RandomClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomResizedCrop,
    Div255,
    Permute
)

from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip
)
from tqdm import tqdm
import datetime
import os

import sys
sys.path.insert(1, '../src')
from random_temporal_subsample import RandomTemporalSubsample
from vit import Embedding, CrossSelfDecoder, ViT_Encoder, Masking

import wandb

def download_model_from_run(run_id, artifact_name="model"):
    """
    Downloads a model artifact from a specified W&B run.

    Args:
        run_id (str): The ID of the run from which to download the model artifact.
        artifact_name (str, optional): The name of the model artifact. Defaults to "model".

    Returns:
        tuple: A tuple containing the downloaded artifact and the run configuration.
    """
    api = wandb.Api()

    run = api.run(run_id)
    config = run.config

    artifacts = run.logged_artifacts()
    artifact = None
    for art in artifacts:
        if art.name.split(":")[0] == artifact_name:
            artifact = art

    if artifact is None:
        print(f'Artifact {artifact_name} not found in the used artifacts of this run.')

    return artifact, config

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

def count_parameters(model):
    """
    Counts the number of trainable parameters in a nn.Module.

    Args:
        model (nn.Module): The model to count the parameters of.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate(model, dataloader, loss_fn, cfg):
    """
    Function to validate the model on a given dataloader.

    Args:
        model (tuple): Tuple containing the embedding, encoder, decoder, and masking models.
        dataloader (torch.utils.data.DataLoader): DataLoader object containing the validation data.
        loss_fn (torch.nn.Module): Loss function used to calculate the validation loss.
        cfg (dict): Configuration dictionary containing various parameters.

    Returns:
        float: Mean validation loss.
    """
    embedding, encoder, decoder, masking = model
    embedding.eval(), encoder.eval(), decoder.eval()
    val_loss = []
    with torch.no_grad():
        for batch_nr, batch in tqdm(enumerate(dataloader), total=vdata.num_videos//cfg['batch_size'], smoothing=50/vdata.num_videos//cfg['batch_size']):
            batch_size = batch['video'].shape[0] # actual batch size (important for last batch)
            batch = batch['video'].to(device)
            images = batch.view(-1, cfg['channels'], cfg['image_size'], cfg['image_size'])

            # embedding
            embeddings = embedding.embedding(images).view(batch_size, cfg['repeated_sampling_factor']+1, -1, cfg['D'])

            # masking
            first_frame = embeddings[:, 0, :, :]
            future_frames, shuff_idx, skipped_token = masking.mask(embeddings[:, 1:, :, :].reshape(batch_size*cfg['repeated_sampling_factor'], -1, cfg['D']), cfg['mask_ratio'], cfg['mask_type'])

            # encoder
            first_frame = encoder(first_frame)
            future_frames = encoder(future_frames)

            # unmasking
            future_frames = embedding.decoder_embedding(future_frames, cfg['mask_type'], shuff_idx, skipped_token)
            output = decoder(future_frames, first_frame.repeat(1, cfg['repeated_sampling_factor'], 1).view(batch_size*cfg['repeated_sampling_factor'], -1, cfg['D']))

            # unembedding
            output = embedding.unembedding(output)

            # calculate loss
            mask = masking.create_mask(cfg['image_size']//cfg['patch_size'], cfg['mask_type'], shuff_idx, skipped_token, H=cfg['image_size'], W=cfg['image_size'], device=device)
            loss = (loss_fn(output, batch[:, 1:].reshape(-1, cfg['channels'], cfg['image_size'], cfg['image_size'])) * mask).sum() / mask.sum()
            val_loss.append(loss.item() / batch_size)

    embedding.train(), encoder.train(), decoder.train()
    return np.array(val_loss).mean()

def train(model, dataloader, vdataloader, loss_fn, optimizer, sched, cfg, run):
    """
    Trains the model using the provided data and hyperparameters. Saves checkpoints
    every cfg['save_model_every'] epochs and plots example images every cfg['plot_every']
    epochs.

    Args:
        model (tuple): Tuple containing the embedding, encoder, decoder, and masking models.
        dataloader (torch.utils.data.DataLoader): DataLoader for the training data.
        vdataloader (torch.utils.data.DataLoader): DataLoader for the validation data.
        loss_fn (torch.nn.Module): Loss function to calculate the training loss.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
        sched (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for the optimizer.
        cfg (dict): Dictionary containing the hyperparameters and configuration settings.
        run: Object for logging the training progress.

    Returns:
        tuple: Tuple containing the training loss and validation loss for each epoch.
    """
    log_dir = "logs/wandb_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(log_dir)
    os.makedirs(log_dir + "plots/")
    train_loss = []
    val_loss = []
    embedding, encoder, decoder, masking = model
    for epoch in range(cfg['epochs']):
        epoch_loss = []
        for batch in tqdm(dataloader, total=data.num_videos//cfg['batch_size'], smoothing=50/data.num_videos//cfg['batch_size']):

            # batch.shape = (batch_size, repeated_sampled_frames, channels, height, width)
            # change to (batch_size * repeated_sampled_frames, channels, height, width)
            batch_size = batch['video'].shape[0] # actual batch size (important for last batch)
            batch = batch['video'].to(device)
            images = batch.view(-1, cfg['channels'], cfg['image_size'], cfg['image_size'])
            optimizer.zero_grad()
            # embedding
            embeddings = embedding.embedding(images).view(batch_size, cfg['repeated_sampling_factor']+1, -1, cfg['D'])

            # masking
            first_frame = embeddings[:, 0, :, :]
            future_frames, shuff_idx, skipped_token = masking.mask(embeddings[:, 1:, :, :].reshape(batch_size*cfg['repeated_sampling_factor'], -1, cfg['D']), cfg['mask_ratio'], cfg['mask_type'])

            # encoder
            first_frame = encoder(first_frame)
            future_frames = encoder(future_frames)

            # unmasking
            future_frames = embedding.decoder_embedding(future_frames, cfg['mask_type'], shuff_idx, skipped_token)

            output = decoder(future_frames, first_frame.repeat(1, cfg['repeated_sampling_factor'], 1).view(batch_size*cfg['repeated_sampling_factor'], -1, cfg['D']))

            # unembedding
            output = embedding.unembedding(output)

            # calculate loss
            mask = masking.create_mask(cfg['image_size']//cfg['patch_size'], cfg['mask_type'], shuff_idx, skipped_token, H=cfg['image_size'], W=cfg['image_size'], device=device)
            
            # Loss is calculated only on the masked tokens (MAE)
            loss = (loss_fn(output, batch[:, 1:].reshape(-1, cfg['channels'], cfg['image_size'], cfg['image_size'])) * mask).sum() / mask.sum()

            epoch_loss.append(loss.item() / batch_size)
            loss.backward()
            optimizer.step()
        
        if not sched is None:
            sched.step()
        train_loss.append(np.array(epoch_loss).mean())
        val_loss.append(validate(model, vdataloader, loss_fn, cfg))
        print("Epoch", epoch, "Training loss:", train_loss[-1], "Validation loss:", val_loss[-1])
        
        if epoch % cfg['save_model_every'] == 0:
            torch.save(encoder.state_dict(), log_dir + "encoder.pt")
            torch.save(decoder.state_dict(), log_dir + "decoder.pt")
            torch.save(embedding.state_dict(), log_dir + "embedding.pt")
            torch.save(optimizer.state_dict(), log_dir + "optimizer.pt")
            if not sched is None:
                torch.save(sched.state_dict(), log_dir + "scheduler.pt")

            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(log_dir + "encoder.pt")
            artifact.add_file(log_dir + "decoder.pt")
            artifact.add_file(log_dir + "embedding.pt")
            artifact.add_file(log_dir + "optimizer.pt")
            if not sched is None:
                artifact.add_file(log_dir + "scheduler.pt")
            run.log_artifact(artifact)
            print("Model saved")

        if epoch % cfg['plot_every'] == 0:
            # plot image
            fig, ax = plt.subplots(5, 4)
            
            for i in range(5):
                ax[i, 0].imshow(unnormalize(batch[i, 0, :, :, :].detach().cpu()).permute(1, 2, 0).numpy())
                ax[i, 1].imshow(unnormalize(batch[i, 1, :, :, :].detach().cpu()).permute(1, 2, 0).numpy())
                ax[i, 2].imshow(unnormalize(output[i*cfg['repeated_sampling_factor'], :, :, :].detach().cpu()).permute(1, 2, 0).numpy())
                ax[i, 3].imshow(mask[0, :, :].permute(1, 2, 0).detach().cpu().numpy())
            plt.savefig(log_dir + f'plots/{epoch}.png', bbox_inches='tight')
            plt.close()
            wandb.log({"example": wandb.Image(log_dir + f'plots/{epoch}.png')}, commit=False)
        wandb.log(
            {
                "train/loss": train_loss[-1],
                "val/loss": val_loss[-1]
            }
        )
    
    torch.save(encoder.state_dict(), log_dir + "encoder.pt")
    torch.save(decoder.state_dict(), log_dir + "decoder.pt")
    torch.save(embedding.state_dict(), log_dir + "embedding.pt")
    torch.save(optimizer.state_dict(), log_dir + "optimizer.pt")
    if not sched is None:
        torch.save(sched.state_dict(), log_dir + "scheduler.pt")

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(log_dir + "encoder.pt")
    artifact.add_file(log_dir + "decoder.pt")
    artifact.add_file(log_dir + "embedding.pt")
    artifact.add_file(log_dir + "optimizer.pt")
    if not sched is None:
        artifact.add_file(log_dir + "scheduler.pt")
    run.log_artifact(artifact)
    print("Model saved")

    return train_loss, val_loss


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

if __name__ == "__main__":
    # Hyperparameters
    cfg = {
        "batch_size": 16,
        "num_workers": 78,
        "channels": 3,
        "image_size": 224,
        "repeated_sampling_factor": 2,
        "lr": 8e-5,
        "weight_decay": 0.05,
        "beta1": 0.9,
        "beta2": 0.95,
        "epochs": 20,
        "data_path": "frames_40/class1/",
        "val_data_path": "frames_40_val/class1/",
        "patch_size": 16,
        "D": 768,
        "encoder_heads": 8,
        "encoder_layers": 12,
        "encoder_mlp_dim": 2048,
        "decoder_heads": 8,
        "decoder_layers": 12,
        "decoder_mlp_dim": 2048,
        "mlp_activation": nn.GELU(),
        "mask_ratio": 0.95,
        "mask_type": 'random',
        "frame_gap_range": (2, 25),
        "fps": 14,
        "use_pretrained": True,
        "pretrained_path": "aweers/dd2412-exploration/ckenfh0b",
        "save_model_every": 4,
        "plot_every": 1,
        "pure_cross_attention": False,
        "use_scheduler": False
    }
    clip_duration = cfg['frame_gap_range'][1] / cfg['fps'] + 0.0001

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print("Using device =", device)

    transform = Compose(
        [
        ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                RandomTemporalSubsample(cfg['frame_gap_range'][0], cfg['frame_gap_range'][1], repeated_sampling=cfg['repeated_sampling_factor']),
                RandomResizedCrop(cfg['image_size'], cfg['image_size'], scale=(1.0, 1.0), aspect_ratio=(1.0, 1.0), interpolation='bilinear'),
                Div255(),
                # mean and std from ImageNet
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # Permute to get Frames, Channel, Height, Width
                Permute((1, 0, 2, 3)),
                RandomHorizontalFlip(p=0.5)
                ]
            ),
            ),
        ]
    )

    label = {"category": "example"} # label is not used, but necessary for the LabeledVideoDataset
    data_list = create_label_list(cfg['data_path'], label)
    vdata_list = create_label_list(cfg['val_data_path'], label)

    data = LabeledVideoDataset(data_list, clip_sampler=RandomClipSampler(clip_duration=clip_duration), decode_audio=False, transform=transform)
    vdata = LabeledVideoDataset(vdata_list, clip_sampler=RandomClipSampler(clip_duration=clip_duration), decode_audio=False, transform=transform)
    cfg['num_videos'] = data.num_videos
    cfg['num_val_videos'] = vdata.num_videos

    dloader = torch.utils.data.DataLoader(
        data,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    vdloader = torch.utils.data.DataLoader(
        vdata,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )

    encoder = ViT_Encoder(cfg['D'], cfg['encoder_heads'], (cfg['encoder_mlp_dim'], cfg['mlp_activation']), cfg['encoder_layers'])
    decoder = CrossSelfDecoder(cfg['D'], cfg['decoder_heads'], (cfg['decoder_mlp_dim'], cfg['mlp_activation']), cfg['decoder_layers'], cfg['pure_cross_attention'])
    masking = Masking(cfg['D'])
    embedding = Embedding(cfg['D'], cfg['patch_size'], cfg['channels'], cfg['image_size']//cfg['patch_size'] * cfg['image_size']//cfg['patch_size'], masking)

    if cfg['use_pretrained']:
        artifact, _ = download_model_from_run(cfg['pretrained_path'])
        artifact_dir = artifact.download()
        
        encoder.load_state_dict(torch.load(artifact_dir + "/encoder.pt", map_location=device))
        decoder.load_state_dict(torch.load(artifact_dir + "/decoder.pt", map_location=device))
        embedding.load_state_dict(torch.load(artifact_dir + "/embedding.pt", map_location=device))

    embedding.train(), encoder.train(), decoder.train()

    embedding.to(device)
    encoder.to(device)
    decoder.to(device)

    embedding_params = count_parameters(embedding)
    encoder_params = count_parameters(encoder)
    decoder_params = count_parameters(decoder)

    model = (embedding, encoder, decoder, masking)

    print("Embedding parameters:", embedding_params)
    print("Encoder parameters:", encoder_params)
    print("Decoder parameters:", decoder_params)
    print("Total parameters:", embedding_params + encoder_params + decoder_params)

    cfg['embedding_params'] = embedding_params
    cfg['encoder_params'] = encoder_params
    cfg['decoder_params'] = decoder_params

    optimizer = torch.optim.AdamW(list(embedding.parameters()) + list(encoder.parameters()) + list(decoder.parameters()), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']), weight_decay=cfg['weight_decay'])
    if cfg['use_scheduler']:
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=1e-6)
    else:
        sched = None
    if cfg['use_pretrained']:
        if os.path.isfile(artifact_dir + "/optimizer.pt"):
            optimizer.load_state_dict(torch.load(artifact_dir + "/optimizer.pt", map_location=device))
        else:
            print("No optimizer found")
        if cfg['use_scheduler'] and os.path.isfile(artifact_dir + "/scheduler.pt"):
            sched.load_state_dict(torch.load(artifact_dir + "/scheduler.pt", map_location=device))
        else:
            print("No scheduler found")

    run = wandb.init(
        project="dd2412-exploration",
        entity="aweers",
        config=cfg
    )

    train_loss, val_loss = train(model, dloader, vdloader, nn.MSELoss(reduction='none'), optimizer, sched, cfg, run)
    run.finish()