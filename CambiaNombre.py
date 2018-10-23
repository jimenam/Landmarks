# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:08:18 2018

@author: jimena
"""
import sys, os
from PIL import Image

def ChangeName(pLandmark, nLandmark, image_dir):
  countTot = 0
  for item in os.listdir(image_dir):
          countTot = countTot +1
          name = item
          name = name.replace(pLandmark, nLandmark)
          os.rename(image_dir + item, image_dir + name)
                    
  print(' TOTAL IMAGENES: ', countTot)

    
def Run():
  if len(sys.argv) != 4 :
    print('Syntax: %s <pLandmark> <nLandmark> <image_dir/>' % sys.argv[0])
    sys.exit(0)
  (pLandmark, nLandmark, image_dir) = sys.argv[1:]

  ChangeName(pLandmark, nLandmark, image_dir)
  
if __name__ == '__main__':
  Run()