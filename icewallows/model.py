import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np
import os

class LabelSmoothedCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothedCE, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(pred, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class ChessTransformerEncoderFT(nn.Module):
    def __init__(self, vocab_sizes, d_model, n_heads, d_queries, d_values, d_inner, n_layers, dropout):
        super(ChessTransformerEncoderFT, self).__init__()
        self.embedding = nn.Embedding(vocab_sizes['moves'], d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_inner, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.fc_out = nn.Linear(d_model, vocab_sizes['moves'])

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.model = ChessTransformerEncoderFT(
            vocab_sizes=config['VOCAB_SIZES'],
            d_model=config['D_MODEL'],
            n_heads=config['N_HEADS'],
            d_queries=config['D_QUERIES'],
            d_values=config['D_VALUES'],
            d_inner=config['D_INNER'],
            n_layers=config['N_LAYERS'],
            dropout=config['DROPOUT']
        )
        self.criterion = LabelSmoothedCE(smoothing=config['LABEL_SMOOTHING'])

    def forward(self, x):
        return self.model(x)

    def score(self, pgn, move):
        # Dummy implementation for scoring a move based on PGN
        # This should be replaced with actual logic
        return np.random.rand()

def load_model(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = Model(config)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    return model

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'model_config.yaml')
    checkpoint_path = os.path.join(current_dir, 'checkpoint.pt')
    checkpoint_path = 'checkpoint.pt'
    model = load_model(config_path, checkpoint_path)
    print("Model loaded successfully.")