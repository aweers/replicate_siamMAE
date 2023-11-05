import torch

def random_temporal_subsample(video, min_gap, max_gap, temporal_dim=-3):
    # Return first and randomly sampled frame
    t = video.shape[temporal_dim]

    # Randomly sample a gap size
    gap = torch.randint(min_gap, max_gap, (1,))
    
    # return tensor with first and randomly sampled frame
    return torch.index_select(video, temporal_dim, torch.tensor([0, gap]))

class RandomTemporalSubsample(torch.nn.Module):
    def __init__(self, min_gap, max_gap, temporal_dim=-3):
        super().__init__()
        self.min_gap = min_gap
        self.max_gap = max_gap
        self._temporal_dim = temporal_dim

    def forward(self, x):
        return random_temporal_subsample(
            x, self.min_gap, self.max_gap, self._temporal_dim
        )