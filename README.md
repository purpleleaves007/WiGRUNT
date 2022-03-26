# WiGRUNT code
Note: Our paper has benn accepted by IEEE Transactions on Human-Machine Systems, the password of the code is sac123.

This project contains two folders, matlab and python, respectively. The former is used to preprocess and visualize CSI data, and the latter is our Dual-Attention CSI Network (DACN) for gesture recognition, and please change the pretrained to 'True' of the ResNet model. If you have any question, please refer to zhangxianghfut@gmail.com or zhangxiang@mail.hfut.edu.cn.

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

### Feature set
We also propose the feture set we used(image), can download in Google Drive:https://drive.google.com/file/d/1fTVv5p0NOQOkKC13Vxt-orYliJEaybrE/view?usp=sharing or Baidu Disk: https://pan.baidu.com/s/1463ZdleseB1wHxcb9bhPxQ 
extract code：rlkb
file example:a-b-c-d-e.jpg
a:user b:gesture c:position d:orientation e:repeat times 
