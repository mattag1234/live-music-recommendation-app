"""
autoencoder.py — neural network architecture for the music recommender.

A feedforward autoencoder that compresses 24 audio-feature dimensions
into a 16-dimensional latent embedding. The encoder half is used at
inference time to produce song fingerprints; cosine similarity over
those fingerprints drives the recommendation.
"""

import torch
import torch.nn as nn

INPUT_DIM = 24          # 12 continuous features + 12 one-hot encoded 'key' features
HIDDEN_DIM = 64         # Size of hidden layers in the encoder/decoder
HIDDEN_DIM_2 = 32       # Size of second hidden layer in the encoder/decoder
LATENT_DIM = 16         # Size of the latent embedding bottleneck  & the song embedding size

class MusicAutoencoder(nn.Module):
    """
    Encoder: 24 -> 64 -> 32 -> 16 (latent embedding)
    Decoder: 16 -> 32 -> 64 -> 24 (reconstruction)

    The encoder is trained with MSE loss on the input features, so it learns to compress the audio features
    Then during inference time, only the encoder is used to produce the 16-dimensional song embeddings, 
    which are compared with cosine similarity for recommendations. This is the generalistic approach , 
    especially for a first pass, but you could experiment with more complex architectures (e.g. variational autoencoder, 
    or adding regularization) if you want to get fancy. (Just be mindful of overfitting, especially with a small dataset and a powerful model.)


    Its super cool, as after learning and coding a small model, we are now implementing an autoencoder
    that can learn to compress and reconstruct song features, 
    and then use that compressed representation for recommendations. 
    This is a great example of how deep learning can be applied to real-world problems like music recommendation!
    """

    def __init__(self):
        super().__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, LATENT_DIM)) # No activation on the output of the encoder (the latent embedding)

        # Note: NO activation after the final layer of the encoder.
        # We want the latent embedding to be unconstrained (any real number).

        # Decoder layers

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM_2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, INPUT_DIM))   # no activation — output must be unconstrained
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encoder-only forward pass — produces the song embedding.
        Used at inference time to generate the fingerprint for similarity search.
        
        Args:
            x: tensor of shape (batch_size, 24)
        Returns:
            embedding tensor of shape (batch_size, 16)
        """
        return self.encoder(x)
    