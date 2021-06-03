#!/bin/bash

sudo pip3 install --force-reinstall opencv-python==3.4.10.35 tqdm websockets
sudo pip3 install --force-reinstall gdown

gdown http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
gdown http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

tar -h -vxf ./VOCtrainval_06-Nov-2007.tar -C ./data
tar -h -vxf ./VOCtest_06-Nov-2007.tar -C ./data
