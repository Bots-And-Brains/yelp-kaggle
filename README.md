YELP - KAGGLE COMPETITION
=========================

#####VERSION 1

This is the version that I used to submit to the yelp kaggle competition in April of 2016.
When I first started on it, in January 2016, Tensorflow had just been released (0.5) a month or so before
and there were still a lot of bugs and little documentation. Now in MArch 2017, there are tools that would
have made it a lot easier. My plan is to refactor this code very soon and make it the base for my talk at Drupalcon
2017.

I also want to create more of a proper report on the things I tried that worked and didn't work as well.
Even see if I can't get an even better result after a year of leaning a lot more.

######VERSION 2

Coming soon..


INSTALLING
===========


Standard folder structure:
-------------------------
- ./yelp-kaggle - (root folder. name doesn't matter)
- ./yelp-kaggle/src - This repo's source code
- ./yelp-kaggle/data - The downloaded data
- ./yelp-kaggle/data/train_photos - Training photos
- ./yelp-kaggle/data/test_photos - Test photos

Installing dependencies
-----------------------

### UBUNTU 14.04

#### iPython notebooks




#### OpenCV
```
conda install opencv
sudo apt-get install python-opencv
wget -O - https://raw.githubusercontent.com/jayrambhia/Install-OpenCV/master/Ubuntu/2.4/opencv2_4_9.sh | bash
```

#### Caffe

https://github.com/tiangolo/caffe/blob/ubuntu-tutorial-b/docs/install_apt2.md - not official, but seems to have almost everything I needed.

Followed instructions, but had issue with libpng during compile. Needed..

From http://stackoverflow.com/questions/32405035/caffe-installation-opencv-libpng16-so-16-linkage-issues
```
cd /usr/lib/x86_64-linux-gnu
sudo ln -s ~/anaconda/lib/libpng16.so.16 libpng16.so.16
sudo ldconfig
```
#### Caffe Feature Extractor

##### Creating features
python caffe_feature_extractor.py -i ../data/images.txt -o ../data/features.txt

