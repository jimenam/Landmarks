# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 17:37:47 2018

@author: jimena
"""

from PIL import Image, ImageOps

desired_size = 512
im_pth = "D:/Master/RomanColiseum_cf898bb7-077b-4426-b678-535e6a433a1d.jpg"

im = Image.open(im_pth)
old_size = im.size

ratio = float(desired_size)/max(old_size)
new_size = tuple([int(x*ratio) for x in old_size])
im = im.resize(new_size, Image.LANCZOS)
new_im = Image.new("RGB", (desired_size, desired_size))
new_im.paste(im, ((desired_size-new_size[0])//2,
                    (desired_size-new_size[1])//2))
new_im.save("D:/Master/NewRomanColiseum_cf898bb7-077b-4426-b678-535e6a433a1d.jpg")
new_im.show()
