# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:08:18 2018

@author: jimena
"""
import sys, os
from PIL import Image
import random

def SplitData(out_dir, image_dir):
  # Define data directories
  train_dir = os.path.join(out_dir, 'train//')
  test_dir = os.path.join(out_dir, 'test//')
  dev_dir = os.path.join(out_dir, 'dev//')
  # Get filenames
  filenames = os.listdir(image_dir)
  filenames = [f for f in filenames]
  # Before splits shuffles elements with a fixed seed so that the split is reproducible
  random.seed(230)
  filenames.sort()
  random.shuffle(filenames)
  # Split images in 80% train, 10% dev and 10% test
  split_1 = int(0.8 * len(filenames))
  split_2 = int(0.9 * len(filenames))
  train_filenames = filenames[:split_1]
  dev_filenames = filenames[split_1:split_2]
  test_filenames = filenames[split_2:]
  for train in train_filenames:
      CopyImage(train_dir, image_dir, train)
  for dev in dev_filenames:
      CopyImage(dev_dir, image_dir, dev)
  for test in test_filenames:
      CopyImage(test_dir, image_dir, test)
          
  print(' TERMINO ')

def CopyImage(out_dir, image_dir, img):
    image = Image.open(os.path.join(image_dir, img))
    image.save(os.path.join(out_dir, img))    
    
def Run():
  (out_dir, image_dir) = sys.argv[1:]

  
  SplitData(out_dir, image_dir)
  
if __name__ == '__main__':
  Run()