# WiGRUNT code
Note: Before the paper is accepted, please send an email to zhangxianghfut@gmail.com or zhangxiang@mail.hfut.edu.cn to obtain the decompression password.

This project contains two folders, matlab and python, respectively. The former is used to preprocess and visualize CSI data, and the latter is our Dual-Attention CSI Network (DACN) for gesture recognition.

### matlab
For CSI visualization

Run QFM.m first to visualize CSI data from each TR pair to iamge first

Run StiFM.m to combine the image of each TR pair 

### python
Our Dual-Attention CSI Network

Run DACN.py to train and test network

da_att.py: attention modules

DACN.py: training the network

data-loader.py: dataset division

options.py: training options 

val.py: gesture recognition with the trained network 

### WiDar3 dataset
http://tns.thss.tsinghua.edu.cn/widar3.0/
