# Faster-Rcnn in Pytorch
- An implementation of [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) in PyTorch.
- Only python source. These is no need to compile nms and roialign cpp
- Comment in many functions

## Prepare install
- cd faster_rcnn_pytorch
- ./install_data.sh

## Run
- python3 ./train.py --cuda True
- python3 ./train.py --cuda True --resume True
- python3 ./infer.py --cuda True

##
You can compare faster_rcnn_pytorch with this
- https://www.mediafire.com/file/m5wx06gmtqvu1km/simple-faster-rcnn.tar.gz

## Refer to
- https://github.com/chenyuntc/simple-faster-rcnn-pytorch
- https://github.com/potterhsu/easy-faster-rcnn.pytorch
