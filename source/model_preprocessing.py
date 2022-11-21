# coding: UTF-8
# developer : TachibanaET 
import os
import sys
import urllib.request
import tarfile
import subprocess
from subprocess import PIPE
import tensorflow as tf
from pprint import pprint

import torch
from torch import nn 

class model_preprocessing:
  def __init__(self):
    self.gpt2_japanese_model_download_url = "https://www.nama.ne.jp/models/gpt2ja-medium.tar.bz2"
    self.models_save_dir = "models/gpt2-pytorch-model-medium"
    self.save_name = "gpt2ja-medium.tar.bz2"
    self.model_path = os.path.join(self.models_save_dir, self.save_name)

  def download_model(self):
    urllib.request.urlretrieve(self.gpt2_japanese_model_download_url, self.model_path)
    print('download_model ... done')

  def unzip_model(self):
    with tarfile.open(self.model_path) as tar:
      def is_within_directory(directory, target):
          
          abs_directory = os.path.abspath(directory)
          abs_target = os.path.abspath(target)
      
          prefix = os.path.commonprefix([abs_directory, abs_target])
          
          return prefix == abs_directory
      
      def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
      
          for member in tar.getmembers():
              member_path = os.path.join(path, member.name)
              if not is_within_directory(path, member_path):
                  raise Exception("Attempted Path Traversal in Tar File")
      
          tar.extractall(path, members, numeric_owner=numeric_owner) 
          
      
      safe_extract(tar, self.models_save_dir)

    print('unzip_model ... done')

  def convert_tf_to_torch(self):
    tf_path = self.models_save_dir + '/' + f'gpt2ja-medium/model-10410000'
    command = f'''
      transformers-cli convert --model_type gpt2 \
      --tf_checkpoint {tf_path} \
      --pytorch_dump_output {self.models_save_dir} \
      --config {self.models_save_dir}/config.json
    '''

    subprocess.run(command, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    print('convert_tf_to_torch ... done')

  def clean_files(self):
    subprocess.run(f'rm -rf {self.models_save_dir}/gpt2ja-medium', shell=True, stdout=sys.stdout, stderr=sys.stderr)
    subprocess.run(f'rm -rf {self.models_save_dir}/{self.save_name}', shell=True, stdout=sys.stdout, stderr=sys.stderr)
    print('clean files ... done')

if __name__ == '__main__':
  prep = model_preprocessing()
  prep.download_model()
  prep.unzip_model()
  prep.convert_tf_to_torch()
  prep.clean_files()