#####################
# UNDER DEVELOPMENT #
#####################

import torch
import torch.nn as nn
from torch.utils.data import DistributedSampler
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
import wandb
from tqdm import tqdm
import time
import os

from random_temporal_subsample import RandomTemporalSubsample
from vit import Embedding, CrossSelfDecoder, ViT_Encoder, Masking

BATCH_SIZE = 64
NUM_WORKERS = 0
CHANNELS = 3
IMAGE_SIZE = 20
REPEATED_SAMPLING_FACTOR = 4
PASS_REPEATED_SAMPLES = False
LR = 3e-4
WEIGHT_DECAY = 0.05
BETA1 = 0.9
BETA2 = 0.95
SWITCH_QUERY_KEY = False
MASK_LOSS = True

# ViT hyperparameters
patch_size = 6
D = 192
encoder_heads = 8
encoder_layers = 4
encoder_mlp_dim = 128
decoder_heads = 8
decoder_layers = 4
decoder_mlp_dim = 128
mlp_activation = nn.GELU()

# masking
mask_ratio = 0.9
mask_type = 'random'

# other hyperparameters
frame_gap_range = (4, 49)
fps = 29

USE_PRETRAINED = False
PRETRAINED_PATH = "logs/output_1700231429/checkpoints/epoch_195"

cfg = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "channels": CHANNELS,
    "image_size": IMAGE_SIZE,
    "repeated_sampling_factor": REPEATED_SAMPLING_FACTOR,
    "lr": LR,
    "weight_decay": WEIGHT_DECAY,
    "switch_query_key": SWITCH_QUERY_KEY,
    "mask_loss": MASK_LOSS,
    "patch_size": patch_size,
    "D": D,
    "encoder_heads": encoder_heads,
    "encoder_layers": encoder_layers,
    "encoder_mlp_dim": encoder_mlp_dim,
    "decoder_heads": decoder_heads,
    "decoder_layers": decoder_layers,
    "decoder_mlp_dim": decoder_mlp_dim,
    "mlp_activation": mlp_activation,
    "mask_ratio": mask_ratio,
    "mask_type": mask_type,
    "frame_gap_range": frame_gap_range,
    "fps": fps,
    "use_pretrained": USE_PRETRAINED,
    "pretrained_path": PRETRAINED_PATH
}

wandb.init(
    project="dd2412-exploration",
    entity="aweers",
    config=cfg
)


# Calculated parameters
clip_duration = frame_gap_range[1] / fps + 0.0001

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
            RandomTemporalSubsample(frame_gap_range[0], frame_gap_range[1], repeated_sampling=REPEATED_SAMPLING_FACTOR),
            RandomResizedCrop(IMAGE_SIZE, IMAGE_SIZE, scale=(0.2, 1.0), aspect_ratio=(1.0, 1.0), interpolation='bilinear'),
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

def unnormalize(tensor):
    return tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

data = Kinetics("data/", clip_sampler=RandomClipSampler(clip_duration=clip_duration), decode_audio=False, transform=transform)
dloader = torch.utils.data.DataLoader(
    data,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    pin_memory=True
)
wandb.config.update({"n_videos": data.num_videos})
cfg["n_videos"] = data.num_videos

# Create ViT
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

embedding = Embedding(D, patch_size, CHANNELS, IMAGE_SIZE//patch_size * IMAGE_SIZE//patch_size)
encoder = ViT_Encoder(D, encoder_heads, (encoder_mlp_dim, mlp_activation), encoder_layers)
decoder = CrossSelfDecoder(D, decoder_heads, (decoder_mlp_dim, mlp_activation), decoder_layers)
masking = Masking(D)

if USE_PRETRAINED:
    embedding.load_state_dict(torch.load(PRETRAINED_PATH + "/embedding.pt"))
    encoder.load_state_dict(torch.load(PRETRAINED_PATH + "/encoder.pt"))
    decoder.load_state_dict(torch.load(PRETRAINED_PATH + "/decoder.pt"))
    masking.load_state_dict(torch.load(PRETRAINED_PATH + "/masking.pt"))

embedding.train(), encoder.train(), decoder.train(), masking.train()

embedding.to(device)
encoder.to(device)
decoder.to(device)
masking.to(device)

embedding_params = count_parameters(embedding)
encoder_params = count_parameters(encoder)
decoder_params = count_parameters(decoder)
masking_params = count_parameters(masking)

model = (embedding, encoder, decoder, masking)

print("Embedding parameters:", embedding_params)
print("Encoder parameters:", encoder_params)
print("Decoder parameters:", decoder_params)
print("Masking parameters:", masking_params)
print("Total parameters:", embedding_params + encoder_params + decoder_params + masking_params)

cfg["embedding_params"] = embedding_params
cfg["encoder_params"] = encoder_params
cfg["decoder_params"] = decoder_params
cfg["masking_params"] = masking_params
cfg["total_params"] = embedding_params + encoder_params + decoder_params + masking_params

def validate(model, dataloader, loss_fn):
    return -1.0

DEBUG = False
def train(model, dataloader, vdataloader, loss_fn, optimizer, epochs):
    # Setup reference batch for qualitative evaluation
    ref_batch = None
    ref_batch_size = min(5, BATCH_SIZE)
    ref_shuff_idx = None
    ref_skipped_token = None
    for b in dataloader:
        ref_batch = b['video'][:ref_batch_size].to(device)
        break

    train_loss = []
    val_loss = []
    embedding, encoder, decoder, masking = model
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_nr, batch in tqdm(enumerate(dataloader)):
            # batch.shape = (batch_size, repeated_sampled_frames, channels, height, width)
            # change to (batch_size * repeated_sampled_frames, channels, height, width)
            batch_size = batch['video'].shape[0] # actual batch size (important for last batch)
            batch = batch['video'].to(device)
            iters = REPEATED_SAMPLING_FACTOR if PASS_REPEATED_SAMPLES else 1
            for _ in range(iters):
                images = batch.view(-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
                if DEBUG: print("images.shape: ", images.shape)
                optimizer.zero_grad()
                # embedding
                embeddings = embedding.embedding(images).view(batch_size, REPEATED_SAMPLING_FACTOR+1, -1, D)
                #print(embeddings[0, 0, 0, :5])
                if DEBUG: print("Embeddings.shape: ", embeddings.shape)

                # masking
                first_frame = embeddings[:, 0, :, :]
                if DEBUG: print("First_frame.shape: ", first_frame.shape)
                future_frames, shuff_idx, skipped_token = masking.mask(embeddings[:, 1:, :, :].reshape(batch_size*REPEATED_SAMPLING_FACTOR, -1, D), mask_ratio, mask_type)
                if DEBUG: print("Future_frames.shape: ", future_frames.shape)

                # encoder
                first_frame = encoder(first_frame)
                if DEBUG: print("First_frame.shape (after encoding): ", first_frame.shape)
                future_frames = encoder(future_frames)
                #print(future_frames[0, 0, :5])
                if DEBUG: print("Future_frames.shape (after encoding): ", future_frames.shape)

                # unmasking
                future_frames = masking.unmask(future_frames, mask_type, shuff_idx, skipped_token)
                if DEBUG: print("Future_frames.shape (after unmasking): ", future_frames.shape)

                # decoder (future_frames is the query, first_frame provide key and value)
                # future_frames.shape = (batch_size * repeated_sampled_frames, N, D)
                # first_frame.shape = (batch_size, N, D)
                
                if SWITCH_QUERY_KEY:
                    output = decoder(first_frame.repeat(1, REPEATED_SAMPLING_FACTOR, 1).view(batch_size*REPEATED_SAMPLING_FACTOR, -1, D), future_frames)
                else:
                    # SiamMAE
                    output = decoder(future_frames, first_frame.repeat(1, REPEATED_SAMPLING_FACTOR, 1).view(batch_size*REPEATED_SAMPLING_FACTOR, -1, D))
                if DEBUG: print("Output.shape: ", output.shape)
                #print(output[0, 0, :5])

                # unembedding
                output = embedding.unembedding(output)
                if DEBUG: print("Output.shape (after unembedding): ", output.shape)

                # calculate loss
                #loss = loss_fn(output.reshape(-1, 1), batch[:, 1:, :, :, :].reshape(-1, 1))
                mask = masking.create_mask(IMAGE_SIZE//patch_size, mask_type, shuff_idx, skipped_token, H=IMAGE_SIZE, W=IMAGE_SIZE, device=device)
                
                if MASK_LOSS:
                    # Loss is calculated only on the masked tokens (MAE)
                    loss = (loss_fn(output, batch[:, 1:].reshape(-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)) * mask).sum() / mask.sum()
                else:
                    # Loss is calculated on all tokens (SiamMAE?)
                    loss = loss_fn(output, batch[:, 1:].reshape(-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)).mean()

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

        train_loss.append(epoch_loss/(data.num_videos*(iters/BATCH_SIZE)))
        val_loss.append(validate(model, vdataloader, loss_fn))
        print("Epoch", epoch, "Training loss:", train_loss[-1], "Validation loss:", val_loss[-1])
        
        if epoch % 5 == 0:
            with torch.no_grad():
                images = ref_batch.view(-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
                embeddings = embedding.embedding(images).view(ref_batch_size, REPEATED_SAMPLING_FACTOR+1, -1, D)
                first_frame = embeddings[:, 0, :, :]
                future_frames = embeddings[:, 1:, :, :].reshape(ref_batch_size*REPEATED_SAMPLING_FACTOR, -1, D)
                
                if ref_shuff_idx is None:
                    future_frames, ref_shuff_idx, ref_skipped_token = masking.mask(future_frames, mask_ratio, mask_type)
                else:
                    future_frames = masking.mask_deterministic(future_frames, ref_shuff_idx, ref_skipped_token)
                
                first_frame = encoder(first_frame)
                future_frames = encoder(future_frames)
                future_frames = masking.unmask(future_frames, mask_type, ref_shuff_idx, ref_skipped_token)
                if SWITCH_QUERY_KEY:
                    output = decoder(future_frames, first_frame.repeat(1, REPEATED_SAMPLING_FACTOR, 1).view(ref_batch_size*REPEATED_SAMPLING_FACTOR, -1, D))
                else:
                    output = decoder(first_frame.repeat(1, REPEATED_SAMPLING_FACTOR, 1).view(ref_batch_size*REPEATED_SAMPLING_FACTOR, -1, D), future_frames)
                output = embedding.unembedding(output)
                mask = masking.create_mask(IMAGE_SIZE//patch_size, mask_type, ref_shuff_idx, ref_skipped_token, H=IMAGE_SIZE, W=IMAGE_SIZE, device=device)

                # loss per sample
                loss = (loss_fn(output, ref_batch[:, 1:].reshape(-1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE)) * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3))
 
                #columns=["sample", "first_frame", "second_frame", "mask", "reconstructed_frame", "ref_loss"]
                #test_table = wandb.Table(columns=columns)
                fig, ax = plt.subplots(5, 4)
                for i in range(5):
                    img_first_frame = unnormalize(ref_batch[i, 0, :, :, :].detach().cpu()).permute(1, 2, 0).numpy()
                    img_second_frame = unnormalize(ref_batch[i, 1, :, :, :].detach().cpu()).permute(1, 2, 0).numpy()
                    img_mask = mask[0, :, :].permute(1, 2, 0).detach().cpu().numpy()
                    img_reconstructed_frame = unnormalize(output[i*REPEATED_SAMPLING_FACTOR, :, :, :].detach().cpu()).permute(1, 2, 0).numpy()

                    #test_table.add_data(i,
                    #    wandb.Image(img_first_frame), 
                    #    wandb.Image(img_second_frame), 
                    #    wandb.Image(img_mask), 
                    #    wandb.Image(img_reconstructed_frame), 
                    #    loss[i].item()
                    #)
                    ax[i, 0].imshow(img_first_frame)
                    ax[i, 1].imshow(img_second_frame)
                    ax[i, 2].imshow(img_mask)
                    ax[i, 3].imshow(img_reconstructed_frame)
                #wandb.log({f"ref_images_epoch_{epoch}": test_table}, commit=False)
                os.mkdir(f"logs/output_{timestamp}/checkpoints/epoch_{epoch}")
                torch.save(embedding.state_dict(), f"logs/output_{timestamp}/checkpoints/epoch_{epoch}/embedding.pt")
                torch.save(encoder.state_dict(), f"logs/output_{timestamp}/checkpoints/epoch_{epoch}/encoder.pt")
                torch.save(decoder.state_dict(), f"logs/output_{timestamp}/checkpoints/epoch_{epoch}/decoder.pt")
                torch.save(masking.state_dict(), f"logs/output_{timestamp}/checkpoints/epoch_{epoch}/masking.pt")
                plt.savefig(f"logs/output_{timestamp}/ref_images/epoch_{epoch}.png")

        wandb.log(
            {
                "train/loss": train_loss[-1],
                "val/loss": val_loss[-1]
            }
        )
            # plot image
            #fig, ax = plt.subplots(3, 5)
            
            #ax[0].imshow(unnormalize(batch[0, 0, :, :, :].detach().cpu()).permute(1, 2, 0).numpy())
            #ax[1].imshow(unnormalize(batch[0, 1, :, :, :].detach().cpu()).permute(1, 2, 0).numpy())
            #ax[2].imshow(unnormalize(output[0, :, :, :].detach().cpu()).permute(1, 2, 0).numpy())
            #ax[3].imshow(mask[0, :, :].permute(1, 2, 0).detach().cpu().numpy())
            #ax[4].plot(train_loss)

            # save image
            #plt.savefig("output_" + timestamp + "/epoch_"+str(epoch)+".png")
    return train_loss, val_loss


optimizer = torch.optim.AdamW(list(embedding.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(masking.parameters()), lr=LR, betas=[BETA1, BETA2], weight_decay=WEIGHT_DECAY)

# create directory for images with unique timestamp
timestamp = str(int(time.time()))
os.mkdir(f"logs/output_{timestamp}")
os.mkdir(f"logs/output_{timestamp}/checkpoints")
os.mkdir(f"logs/output_{timestamp}/ref_images")

# save cfg in directory
with open(f"logs/output_{timestamp}/config.txt", "w") as f:
    f.write(str(cfg))

train_loss, val_loss = train(model, dloader, dloader, nn.MSELoss(reduction='none'), optimizer, 1000)