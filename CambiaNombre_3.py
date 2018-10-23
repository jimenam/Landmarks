# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:08:18 2018

@author: jimena
"""
import sys, os

def ChangeName(image_dir):
  countTot = 0
  for item in os.listdir(image_dir):
          countTot = countTot +1
          name = item
          name = name.replace('Unknown', '17')
          os.rename(image_dir + item, image_dir + name)
                    
  print(' TOTAL IMAGENES: ', countTot)

    
def Run():
  if len(sys.argv) != 2 :
    print('Syntax: <image_dir/>' % sys.argv[0])
    sys.exit(0)
  (image_dir) = sys.argv[1]
  ChangeName(image_dir)
  
if __name__ == '__main__':
  Run()