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
          name = name.replace('Alcatraz', '0')
          name = name.replace('AlhambraPalace', '1')
          name = name.replace('BasilicaofSaintPeter', '2')
          name = name.replace('CasaBatllo', '3')
          name = name.replace('CharlesBridge', '4')
          name = name.replace('Eiffel', '5')
          name = name.replace('HollywoodSign', '6')
          name = name.replace('ItsukushimaShrine', '7')
          name = name.replace('MoulinRouge', '8')
          name = name.replace('NationalArtMuseumofCatalonia', '9')
          name = name.replace('ParkGuell', '10')
          name = name.replace('PetronasTwinTowers', '11')
          name = name.replace('RialtoBridge', '12')
          name = name.replace('RomanColiseum', '13')
          name = name.replace('RomePantheon', '14')
          name = name.replace('StatueofLiberty', '15')
          name = name.replace('Triomphe', '16')
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