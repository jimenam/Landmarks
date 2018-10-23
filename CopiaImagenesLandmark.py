# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:08:18 2018

@author: jimena
"""
import sys, os, csv
from PIL import Image
import uuid

def ParseData(data_file, out_dir, image_dir, landmark_id, landmark_name):
  csvfile = open(data_file, 'r')
  csvreader = csv.reader(csvfile)
  countTot = 0
  for line in csvreader:
      if str(line[2]) == landmark_id:
          countTot = countTot +1
          CopyImage(out_dir, image_dir, line[0], landmark_name)
          
  print(' TOTAL IMAGENES: ', countTot)

def CopyImage(out_dir, image_dir, id, landmark_name):
    filename = os.path.join(image_dir, '%s.jpg' % id)
    if not os.path.exists(filename):
        print(' FILENAME ', filename)    
        return
    image = Image.open(filename)
    nameFile =  landmark_name + str(uuid.uuid4())
    image.save(os.path.join(out_dir, '%s.jpg' % nameFile))    
    os.remove(filename)
    
def Run():
  if len(sys.argv) != 6 :
    print('Syntax: %s <data_file.csv> <output_dir/> <image_dir/> <landmark_id> <landmark_name>' % sys.argv[0])
    sys.exit(0)
  (data_file, out_dir, image_dir, landmark_id, landmark_name) = sys.argv[1:]

  if not os.path.exists(out_dir):
    os.mkdir(out_dir)

  ParseData(data_file, out_dir, image_dir, landmark_id, landmark_name)
  
if __name__ == '__main__':
  Run()