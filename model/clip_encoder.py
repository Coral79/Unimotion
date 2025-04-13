import clip
import os
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time

class Clip_Encoder(nn.Module):
    def __init__(self, clip_version=None, clip_dim=512, latent_dim=512, dataset= 'babel', **kargs):
        super().__init__()
        self.clip_dim = clip_dim
        self.latent_dim = int(latent_dim) 
        self.embed_text = nn.Linear(self.clip_dim, self.latent_dim)
        print('EMBED TEXT')
        print('Loading CLIP...')
        self.clip_version = clip_version
        self.clip_model = self.load_and_freeze_clip(clip_version)
        if torch.cuda.is_available():
            self.clip_model = self.clip_model.to('cuda')
        else:
            print("CUDA is not available. The model will remain on the CPU.")
        self.dataset = dataset



    def load_and_freeze_clip(self,clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu', jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
    
    def encode_text(self, raw_text):
        # raw_text - list (batch_size length) of strings with input text prompts
        # device = next(self.parameters()).device #cpu 
        device = 'cuda'
        # print(device)
        max_text_len = 20 if self.dataset in ['humanml', 'kit'] else None  # Specific hardcoding for humanml dataset
        if max_text_len is not None:
            default_context_length = 77
            context_length = max_text_len + 2 # start_token + 20 + end_token
            assert context_length < default_context_length
            # print(context_length)
            texts = clip.tokenize(raw_text, context_length=context_length, truncate=True).to(device) # [bs, context_length] # if n_tokens > context_length -> will truncate
            # print('texts', texts.shape)
            zero_pad = torch.zeros([texts.shape[0], default_context_length-context_length], dtype=texts.dtype, device=texts.device)
            texts = torch.cat([texts, zero_pad], dim=1)
            # print('texts after pad', texts.shape, texts)
        else:
            texts = clip.tokenize(raw_text, truncate=True).to(device) # [bs, context_length] # if n_tokens > 77 -> will truncate
            # print(len(texts))
        return self.clip_model.encode_text(texts).float()
    
    def forward(self, text, y=None):
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """
        # print(text)
        enc_text = self.encode_text(text)
        # emb = self.embed_text(enc_text)
        return enc_text
