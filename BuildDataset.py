# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:08:18 2018

@author: jimena
"""
import sys, os, csv
from PIL import Image
import uuid

def ParseData(landmark_name, out_dir, image_dir):
  countTot = 0
  for item in os.listdir(image_dir):
          countTot = countTot +1
          image = Image.open(image_dir + item)
          image = image.resize((512, 512), Image.LANCZOS)
          nameFile =  landmark_name + str(uuid.uuid4())
          image.save(os.path.join(out_dir, '%s.jpg' % nameFile))    
          
  print(' TOTAL IMAGENES: ', countTot)
    
    
def Run():
  if len(sys.argv) != 4 :
    print('Syntax: %s <landmark_name> <output_dir/> <image_dir/>' % sys.argv[0])
    sys.exit(0)
  (landmark_name, out_dir, image_dir) = sys.argv[1:]

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  ParseData(landmark_name, out_dir, image_dir)
  
if __name__ == '__main__':
  Run()