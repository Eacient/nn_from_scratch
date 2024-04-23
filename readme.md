# 仓库说明

神经网络与深度学习课程作业 从零实现多层感知机

### 模型训练

下载数据集

```
mkdir ./data
cd ./data
!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
!wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
```

训练脚本

```
import os
import sys
sys.path.append('./src')
from nn_scratch import train_nn

data_dir = '/content/drive/MyDrive/nn_from_scratch/data'
save_dir = '/content/drive/MyDrive/nn_from_scratch/test_train'
os.makedirs(save_dir, exist_ok=True)

train_nn(data_dir, 256, 'relu', 15, 256, 0.1, 0.005, 1, 0.8, save_dir, plot=True)
```

参数搜索脚本

```
from nn_scratch import param_search
data_dir = '/content/drive/MyDrive/nn_from_scratch/data'
root_dir = '/content/drive/MyDrive/nn_from_scratch/test_param_search'

param_search(data_dir, root_dir)
```

### 模型测试

```
from nn_scratch import test_nn
data_dir = '/content/drive/MyDrive/nn_from_scratch/data'
model_path = '/content/drive/MyDrive/nn_from_scratch/test_param_search/256-relu-100-0.100-0.000-0.800/model.pth.npy'

test_nn(data_dir, model_path, 256, 'relu', -1)
```