# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:08:18 2018

@author: jimena
"""
import sys, os, csv
from PIL import Image
import uuid

def ParseData(data_file, image_dir):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  countTot = 0
  for line in csvreader:
      if str(line[2]) == '10728' or str(line[2]) == '11761' or str(line[2]) == '9529' or str(line[2]) == '10400' or str(line[2]) == '4697' or str(line[2]) == '6542' or str(line[2]) == '3789' or str(line[2]) == '7476' or str(line[2]) == '4324' or str(line[2]) == '1358' or str(line[2]) == '10728':
          countTot = countTot +1
          DeleteImage(image_dir, line[0])
          
  print(' TOTAL IMAGENES: ', countTot)

def DeleteImage(image_dir, id):
    filename = os.path.join(image_dir, '%s.jpg' % id)
    if not os.path.exists(filename):
        print(' FILENAME ', filename)    
        return
    os.remove(filename)
    
def Run():
  if len(sys.argv) != 3 :
    print('Syntax: %s <data_file.csv> <image_dir/>' % sys.argv[0])
    sys.exit(0)
  (data_file, image_dir) = sys.argv[1:]

  ParseData(data_file, image_dir)
  
if __name__ == '__main__':
  Run()