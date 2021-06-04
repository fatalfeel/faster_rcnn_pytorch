# Faster-Rcnn in Pytorch
- An implementation of [Faster R-CNN](https://arxiv.org/pdf/1506.01497.pdf) in PyTorch.
- Only python source. These is no need to compile nms and roialign cpp
- Comment in many functions
- You can debug any line in cpu mode

## Prepare install
- cd faster_rcnn_pytorch
- ./install_data.sh

## Run
- python3 ./train.py --cuda True
- python3 ./train.py --cuda True --resume True
- python3 ./infer.py --cuda True

## Performance
- GeForce GTX 1650 4GB
- CUDA Version 10.2
- Train 3 frames per second
- Infer 8 frames per second

##
You can compare with simple-faster-rcnn that can debug any line in cpu mode
- https://www.mediafire.com/file/m5wx06gmtqvu1km/simple-faster-rcnn.tar.gz

## Refer to
- https://github.com/chenyuntc/simple-faster-rcnn-pytorch
- https://github.com/potterhsu/easy-faster-rcnn.pytorch
