import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorchvideo.data import Kinetics, RandomClipSampler
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


def unnormalize(tensor):
    unnormalized = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    return (torch.clamp(unnormalized, 0, 1) * 255).to(dtype=torch.uint8)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate(model, dataloader, loss_fn, cfg):
    embedding, encoder, decoder, masking = model
    embedding.eval(), encoder.eval(), decoder.eval()
    val_loss = []
    with torch.no_grad():
        for batch_nr, batch in enumerate(dataloader):
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
            if batch_nr >= 2: break
    embedding.train(), encoder.train(), decoder.train()
    return np.array(val_loss).mean()

def train(model, dataloader, vdataloader, loss_fn, optimizer, cfg, run):
    DEBUG = False
    log_dir = "logs/wandb_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(log_dir)
    os.makedirs(log_dir + "plots/")
    batch_orig = None
    for b in dataloader:
        batch_orig = b
        break
    train_loss = []
    val_loss = []
    embedding, encoder, decoder, masking = model
    for epoch in range(cfg['epochs']):
        epoch_loss = []
        #for batch_nr, batch in enumerate(dataloader):
        for batch_nr, batch in enumerate(tqdm(dataloader, total=data.num_videos//cfg['batch_size'])):

            # batch.shape = (batch_size, repeated_sampled_frames, channels, height, width)
            # change to (batch_size * repeated_sampled_frames, channels, height, width)
            batch_size = batch['video'].shape[0] # actual batch size (important for last batch)
            batch = batch['video'].to(device)
            images = batch.view(-1, cfg['channels'], cfg['image_size'], cfg['image_size'])
            if DEBUG: print("images.shape: ", images.shape)
            optimizer.zero_grad()
            # embedding
            embeddings = embedding.embedding(images).view(batch_size, cfg['repeated_sampling_factor']+1, -1, cfg['D'])
            #print(embeddings[0, 0, 0, :5])
            if DEBUG: print("Embeddings.shape: ", embeddings.shape)

            # masking
            first_frame = embeddings[:, 0, :, :]
            if DEBUG: print("First_frame.shape: ", first_frame.shape)
            future_frames, shuff_idx, skipped_token = masking.mask(embeddings[:, 1:, :, :].reshape(batch_size*cfg['repeated_sampling_factor'], -1, cfg['D']), cfg['mask_ratio'], cfg['mask_type'])
            if DEBUG: print("Future_frames.shape: ", future_frames.shape)

            # encoder
            first_frame = encoder(first_frame)
            if DEBUG: print("First_frame.shape (after encoding): ", first_frame.shape)
            future_frames = encoder(future_frames)
            #print(future_frames[0, 0, :5])
            if DEBUG: print("Future_frames.shape (after encoding): ", future_frames.shape)

            # unmasking
            future_frames = embedding.decoder_embedding(future_frames, cfg['mask_type'], shuff_idx, skipped_token)
            if DEBUG: print("Future_frames.shape (after unmasking): ", future_frames.shape)

            output = decoder(future_frames, first_frame.repeat(1, cfg['repeated_sampling_factor'], 1).view(batch_size*cfg['repeated_sampling_factor'], -1, cfg['D']))
            #output = decoder(first_frame.repeat(1, REPEATED_SAMPLING_FACTOR, 1).view(batch_size*REPEATED_SAMPLING_FACTOR, -1, D), future_frames)
            if DEBUG: print("Output.shape: ", output.shape)
            #print(output[0, 0, :5])

            # unembedding
            output = embedding.unembedding(output)
            if DEBUG: print("Output.shape (after unembedding): ", output.shape)

            # calculate loss
            #loss = loss_fn(output.reshape(-1, 1), batch[:, 1:, :, :, :].reshape(-1, 1))
            mask = masking.create_mask(cfg['image_size']//cfg['patch_size'], cfg['mask_type'], shuff_idx, skipped_token, H=cfg['image_size'], W=cfg['image_size'], device=device)
            
            # Loss is calculated only on the masked tokens (MAE)
            loss = (loss_fn(output, batch[:, 1:].reshape(-1, cfg['channels'], cfg['image_size'], cfg['image_size'])) * mask).sum() / mask.sum()

            epoch_loss.append(loss.item() / batch_size)
            loss.backward()
            optimizer.step()
            if batch_nr >= 5: break

        train_loss.append(np.array(epoch_loss).mean())
        val_loss.append(validate(model, vdataloader, loss_fn, cfg))
        print("Epoch", epoch, "Training loss:", train_loss[-1], "Validation loss:", val_loss[-1])
        wandb.log(
            {
                "train/loss": train_loss[-1],
                "val/loss": val_loss[-1]
            }
        )
        if epoch % cfg['save_model_every'] == 0:
            torch.save(encoder.state_dict(), log_dir + "encoder.pt")
            torch.save(decoder.state_dict(), log_dir + "decoder.pt")
            torch.save(embedding.state_dict(), log_dir + "embedding.pt")

            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(log_dir + "encoder.pt")
            artifact.add_file(log_dir + "decoder.pt")
            artifact.add_file(log_dir + "embedding.pt")
            run.log_artifact(artifact)
            print("Model saved")

        if epoch % cfg['plot_every'] == 0:
            # plot image
            fig, ax = plt.subplots(3, 4)
            
            for i in range(3):
                ax[i, 0].imshow(unnormalize(batch[i, 0, :, :, :].detach().cpu()).permute(1, 2, 0).numpy())
                ax[i, 1].imshow(unnormalize(batch[i, 1, :, :, :].detach().cpu()).permute(1, 2, 0).numpy())
                ax[i, 2].imshow(unnormalize(output[i*cfg['repeated_sampling_factor'], :, :, :].detach().cpu()).permute(1, 2, 0).numpy())
                ax[i, 3].imshow(mask[0, :, :].permute(1, 2, 0).detach().cpu().numpy())
            plt.savefig(log_dir + f'plots/{epoch}.png', bbox_inches='tight')
            plt.close()
            wandb.log({"example": wandb.Image(log_dir + f'plots/{epoch}.png')})
    return train_loss, val_loss

if __name__ == "__main__":
    cfg = {
        "batch_size": 8,
        "num_workers": 0,
        "channels": 3,
        "image_size": 224,
        "repeated_sampling_factor": 2,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "beta1": 0.9,
        "beta2": 0.95,
        "epochs": 60,
        "data_path": "data_20/",
        "val_data_path": "data_20_val/",
        "patch_size": 16,
        "D": 768,
        "encoder_heads": 8,
        "encoder_layers": 2,#12
        "encoder_mlp_dim": 512,#2048
        "decoder_heads": 8,
        "decoder_layers": 2,#12
        "decoder_mlp_dim": 512,#2048
        "mlp_activation": nn.GELU(),
        "mask_ratio": 0.95,
        "mask_type": 'random',
        "frame_gap_range": (4, 49),
        "fps": 29,
        "use_pretrained": False,
        "pretrained_path": "",
        "save_model_every": 5,
        "plot_every": 1
    }

    # Calculated parameters
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
                RandomResizedCrop(cfg['image_size'], cfg['image_size'], scale=(0.2, 1.0), aspect_ratio=(1.0, 1.0), interpolation='bilinear'),
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

    data = Kinetics(cfg['data_path'], clip_sampler=RandomClipSampler(clip_duration=clip_duration), decode_audio=False, transform=transform)
    vdata = Kinetics(cfg['val_data_path'], clip_sampler=RandomClipSampler(clip_duration=clip_duration), decode_audio=False, transform=transform)
    cfg['num_videos'] = data.num_videos
    cfg['num_val_videos'] = vdata.num_videos

    dloader = torch.utils.data.DataLoader(
        data,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False,
        pin_memory=True
    )
    vdloader = torch.utils.data.DataLoader(
        vdata,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False,
        pin_memory=True
    )

    encoder = ViT_Encoder(cfg['D'], cfg['encoder_heads'], (cfg['encoder_mlp_dim'], cfg['mlp_activation']), cfg['encoder_layers'])
    decoder = CrossSelfDecoder(cfg['D'], cfg['decoder_heads'], (cfg['decoder_mlp_dim'], cfg['mlp_activation']), cfg['decoder_layers'])
    masking = Masking(cfg['D'])
    embedding = Embedding(cfg['D'], cfg['patch_size'], cfg['channels'], cfg['image_size']//cfg['patch_size'] * cfg['image_size']//cfg['patch_size'], masking)

    if cfg['use_pretrained']:
        encoder.load_state_dict(torch.load(cfg['pretrained_path'] + "encoder.pt"))
        decoder.load_state_dict(torch.load(cfg['pretrained_path'] + "decoder.pt"))
        embedding.load_state_dict(torch.load(cfg['pretrained_path'] + "embedding.pt"))

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

    run = wandb.init(
        project="dd2412-exploration",
        entity="aweers",
        config=cfg
    )

    train_loss, val_loss = train(model, dloader, vdloader, nn.MSELoss(reduction='none'), optimizer, cfg, run)
    run.finish()