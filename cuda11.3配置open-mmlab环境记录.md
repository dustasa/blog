部分参考[open-mmlab环境配置](https://blog.csdn.net/weixin_44994838/article/details/123527875 "open-mmlab环境配置")
```shell
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

# 从https://download.pytorch.org/whl/cu113/torch_stable.html下载以下文件
pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio===0.11.0 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 此时运行torch.cuda.is_available()可以看到为true

# 安装 mmcv-full
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# 安装 MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"

# 之后运行官方的验证代码可以运行
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
device = 'cuda:0'
# 初始化检测器
model = init_detector(config_file, checkpoint_file, device=device)
# 推理演示图像
inference_detector(model, 'demo/demo.jpg')

```