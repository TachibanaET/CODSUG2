import json
import os
import numpy as np
import argparse
from tqdm import tqdm
from utility.encode_bpe import BPEEncoder_ja
import torch
import torch.nn.functional as F
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import requests

class GPT2FineTune():
  def __init__(self, h_params, bpe_encoder):
    # print(bpe_encoder)
    self.h_params = h_params
    self.tokenizer = bpe_encoder
    self.pre_trained_model = GPT2LMHeadModel.from_pretrained(self.h_params['model_path'])

  def fine_tune(self):
    torch.backends.cudnn.benchmark = True
    self.pre_trained_model = self.model.to(self.h_params['device'])

    if self.h_params['device'] == 'cuda':
      self.pre_trained_model = torch.nn.DataParallel(self.pre_trained_model)

    self.model.train()
    

if __name__ == '__main__':
  h_params = {
    'temperature' : 1,
    'top_k' : 40,
    'top_p' : 0.9,
    'batch_size' : 64,
    'epochs' : 50,
    'learning_rate' : 1e-4,
    'warmup_steps' : 5000,
    'max_seq_len' : 256,
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'model_path' : '/workspace/source/models/gpt2-pytorch-model-medium/',
  }

  with open('ja-bpe.txt') as f:
    bpe = f.read().split('\n')

  with open('emoji.json') as f:
    emoji = json.loads(f.read())

  bpe_encoder = BPEEncoder_ja(bpe, emoji)
  gpt2_fine_tune = GPT2FineTune(
    h_params = h_params,
    bpe_encoder = bpe_encoder
  )
