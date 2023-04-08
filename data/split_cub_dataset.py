from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import shutil
from shutil import copyfile

# Change the path to your dataset folder:
base_folder = 'CUB_200_2011/'

# These path should be fine
images_txt_path = base_folder+ 'images.txt'
train_test_split_path =  base_folder+ 'train_test_split.txt'
images_path =  base_folder+ 'images/'

# Here declare where you want to place the train/test folders
# You don't need to create them!
test_folder = 'CUB/test/'
train_folder = 'CUB/train/'


def ignore_files(dir,files): return [f for f in files if os.path.isfile(os.path.join(dir,f))]

shutil.copytree(images_path,test_folder,ignore=ignore_files)
shutil.copytree(images_path,train_folder,ignore=ignore_files)

with open(images_txt_path) as f:
  images_lines = f.readlines()

with open(train_test_split_path) as f:
  split_lines = f.readlines()

test_images, train_images = 0,0

for image_line,split_line in zip(images_lines,split_lines):

  image_line = (image_line.strip()).split(' ')
  split_line = (split_line.strip()).split(' ')

  image = plt.imread(images_path + image_line[1])

  # Use only RGB images, avoid grayscale
  if len(image.shape) == 3:

    # If test image
    if(int(split_line[1]) is 0):
      copyfile(images_path+image_line[1],test_folder+image_line[1])
      test_images += 1 
    else:
      # If train image
      copyfile(images_path+image_line[1],train_folder+image_line[1])
      train_images += 1 

print(train_images,test_images)
assert train_images == 5990
assert test_images == 5790

print('Dataset succesfully splitted!')