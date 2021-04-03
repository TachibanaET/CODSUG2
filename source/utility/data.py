from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import pandas as pd 


class FineTuneDataset(Dataset):
  def __init__(self, dataset_path, tokenizer):
    super().__init__()
    
    self.data_list = []
    self.end_of_text_token = '<|endoftext|>'
    self.tokenizer = tokenizer

    df = pd.read_csv(dataset_path, 
                      sep="\t", 
                      usecols=[0,1],
                      names=('pair_0', 'pair_1')
                    )
    
    for index, row in df.iterrows():
      input_str = f"{row['pair_0']}<SEP>{row['pair_1']}{self.end_of_text_token}"
      self.data_list.append(input_str)

    print('loading data set ... done')

  def __len__(self):
    return len(self.data_list)

  def __getitem__(self, item):
    tokenize_result = self.tokenizer.encode( self.review_list[item], padding=True, max_len=256)
    return {
      'ids' : torch.LongTensor(tokenize_result['input_ids']),
      'mask': torch.LongTensor(tokenize_result['attention_mask'])
    }

if __name__ == '__main__':
  