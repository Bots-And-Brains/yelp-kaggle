YELP - KAGGLE COMPETITION
=========================


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
