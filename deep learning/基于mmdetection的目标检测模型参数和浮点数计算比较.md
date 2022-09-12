<details>
<summary>1. RetinaNet_Res 50_fpn_1x_mini_coco</summary>

<pre><code>

RetinaNet(
  37.743 M, 100.000% Params, 239.317 GFLOPs, 100.000% FLOPs, 
  (backbone): ResNet(
    23.283 M, 61.687% Params, 84.077 GFLOPs, 35.132% FLOPs, 
    (conv1): Conv2d(0.0 M, 0.000% Params, 2.408 GFLOPs, 1.006% FLOPs, 3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): BatchNorm2d(0.0 M, 0.000% Params, 0.033 GFLOPs, 0.014% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (relu): ReLU(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.007% FLOPs, inplace=True)
    (maxpool): MaxPool2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.007% FLOPs, kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): ResLayer(
      0.0 M, 0.000% Params, 13.885 GFLOPs, 5.802% FLOPs, 
      (0): Bottleneck(
        0.0 M, 0.000% Params, 4.825 GFLOPs, 2.016% FLOPs, 
        (conv1): Conv2d(0.0 M, 0.000% Params, 0.262 GFLOPs, 0.110% FLOPs, 64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.003% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.0 M, 0.000% Params, 2.359 GFLOPs, 0.986% FLOPs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.003% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.0 M, 0.000% Params, 1.049 GFLOPs, 0.438% FLOPs, 64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.033 GFLOPs, 0.014% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.025 GFLOPs, 0.010% FLOPs, inplace=True)
        (downsample): Sequential(
          0.0 M, 0.000% Params, 1.081 GFLOPs, 0.452% FLOPs, 
          (0): Conv2d(0.0 M, 0.000% Params, 1.049 GFLOPs, 0.438% FLOPs, 64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(0.0 M, 0.000% Params, 0.033 GFLOPs, 0.014% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        0.0 M, 0.000% Params, 4.53 GFLOPs, 1.893% FLOPs, 
        (conv1): Conv2d(0.0 M, 0.000% Params, 1.049 GFLOPs, 0.438% FLOPs, 256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.003% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.0 M, 0.000% Params, 2.359 GFLOPs, 0.986% FLOPs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.003% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.0 M, 0.000% Params, 1.049 GFLOPs, 0.438% FLOPs, 64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.033 GFLOPs, 0.014% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.025 GFLOPs, 0.010% FLOPs, inplace=True)
      )
      (2): Bottleneck(
        0.0 M, 0.000% Params, 4.53 GFLOPs, 1.893% FLOPs, 
        (conv1): Conv2d(0.0 M, 0.000% Params, 1.049 GFLOPs, 0.438% FLOPs, 256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.003% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.0 M, 0.000% Params, 2.359 GFLOPs, 0.986% FLOPs, 64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.003% FLOPs, 64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.0 M, 0.000% Params, 1.049 GFLOPs, 0.438% FLOPs, 64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.0 M, 0.000% Params, 0.033 GFLOPs, 0.014% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.025 GFLOPs, 0.010% FLOPs, inplace=True)
      )
    )
    (layer2): ResLayer(
      1.22 M, 3.231% Params, 21.154 GFLOPs, 8.839% FLOPs, 
      (0): Bottleneck(
        0.379 M, 1.005% Params, 7.674 GFLOPs, 3.207% FLOPs, 
        (conv1): Conv2d(0.033 M, 0.087% Params, 2.097 GFLOPs, 0.876% FLOPs, 256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.0 M, 0.001% Params, 0.016 GFLOPs, 0.007% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.147 M, 0.391% Params, 2.359 GFLOPs, 0.986% FLOPs, 128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.004 GFLOPs, 0.002% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.066 M, 0.174% Params, 1.049 GFLOPs, 0.438% FLOPs, 128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.001 M, 0.003% Params, 0.016 GFLOPs, 0.007% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.018 GFLOPs, 0.008% FLOPs, inplace=True)
        (downsample): Sequential(
          0.132 M, 0.350% Params, 2.114 GFLOPs, 0.883% FLOPs, 
          (0): Conv2d(0.131 M, 0.347% Params, 2.097 GFLOPs, 0.876% FLOPs, 256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(0.001 M, 0.003% Params, 0.016 GFLOPs, 0.007% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        0.28 M, 0.742% Params, 4.493 GFLOPs, 1.878% FLOPs, 
        (conv1): Conv2d(0.066 M, 0.174% Params, 1.049 GFLOPs, 0.438% FLOPs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.0 M, 0.001% Params, 0.004 GFLOPs, 0.002% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.147 M, 0.391% Params, 2.359 GFLOPs, 0.986% FLOPs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.004 GFLOPs, 0.002% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.066 M, 0.174% Params, 1.049 GFLOPs, 0.438% FLOPs, 128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.001 M, 0.003% Params, 0.016 GFLOPs, 0.007% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.012 GFLOPs, 0.005% FLOPs, inplace=True)
      )
      (2): Bottleneck(
        0.28 M, 0.742% Params, 4.493 GFLOPs, 1.878% FLOPs, 
        (conv1): Conv2d(0.066 M, 0.174% Params, 1.049 GFLOPs, 0.438% FLOPs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.0 M, 0.001% Params, 0.004 GFLOPs, 0.002% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.147 M, 0.391% Params, 2.359 GFLOPs, 0.986% FLOPs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.004 GFLOPs, 0.002% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.066 M, 0.174% Params, 1.049 GFLOPs, 0.438% FLOPs, 128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.001 M, 0.003% Params, 0.016 GFLOPs, 0.007% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.012 GFLOPs, 0.005% FLOPs, inplace=True)
      )
      (3): Bottleneck(
        0.28 M, 0.742% Params, 4.493 GFLOPs, 1.878% FLOPs, 
        (conv1): Conv2d(0.066 M, 0.174% Params, 1.049 GFLOPs, 0.438% FLOPs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.0 M, 0.001% Params, 0.004 GFLOPs, 0.002% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.147 M, 0.391% Params, 2.359 GFLOPs, 0.986% FLOPs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.0 M, 0.001% Params, 0.004 GFLOPs, 0.002% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.066 M, 0.174% Params, 1.049 GFLOPs, 0.438% FLOPs, 128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.001 M, 0.003% Params, 0.016 GFLOPs, 0.007% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.012 GFLOPs, 0.005% FLOPs, inplace=True)
      )
    )
    (layer3): ResLayer(
      7.098 M, 18.807% Params, 30.012 GFLOPs, 12.541% FLOPs, 
      (0): Bottleneck(
        1.512 M, 4.007% Params, 7.638 GFLOPs, 3.192% FLOPs, 
        (conv1): Conv2d(0.131 M, 0.347% Params, 2.097 GFLOPs, 0.876% FLOPs, 512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.59 M, 1.563% Params, 2.359 GFLOPs, 0.986% FLOPs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.002 M, 0.005% Params, 0.008 GFLOPs, 0.003% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.009 GFLOPs, 0.004% FLOPs, inplace=True)
        (downsample): Sequential(
          0.526 M, 1.395% Params, 2.105 GFLOPs, 0.880% FLOPs, 
          (0): Conv2d(0.524 M, 1.389% Params, 2.097 GFLOPs, 0.876% FLOPs, 512, 1024, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(0.002 M, 0.005% Params, 0.008 GFLOPs, 0.003% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        1.117 M, 2.960% Params, 4.475 GFLOPs, 1.870% FLOPs, 
        (conv1): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.59 M, 1.563% Params, 2.359 GFLOPs, 0.986% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.002 M, 0.005% Params, 0.008 GFLOPs, 0.003% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.006 GFLOPs, 0.003% FLOPs, inplace=True)
      )
      (2): Bottleneck(
        1.117 M, 2.960% Params, 4.475 GFLOPs, 1.870% FLOPs, 
        (conv1): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.59 M, 1.563% Params, 2.359 GFLOPs, 0.986% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.002 M, 0.005% Params, 0.008 GFLOPs, 0.003% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.006 GFLOPs, 0.003% FLOPs, inplace=True)
      )
      (3): Bottleneck(
        1.117 M, 2.960% Params, 4.475 GFLOPs, 1.870% FLOPs, 
        (conv1): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.59 M, 1.563% Params, 2.359 GFLOPs, 0.986% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.002 M, 0.005% Params, 0.008 GFLOPs, 0.003% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.006 GFLOPs, 0.003% FLOPs, inplace=True)
      )
      (4): Bottleneck(
        1.117 M, 2.960% Params, 4.475 GFLOPs, 1.870% FLOPs, 
        (conv1): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.59 M, 1.563% Params, 2.359 GFLOPs, 0.986% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.002 M, 0.005% Params, 0.008 GFLOPs, 0.003% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.006 GFLOPs, 0.003% FLOPs, inplace=True)
      )
      (5): Bottleneck(
        1.117 M, 2.960% Params, 4.475 GFLOPs, 1.870% FLOPs, 
        (conv1): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(0.59 M, 1.563% Params, 2.359 GFLOPs, 0.986% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.001 M, 0.001% Params, 0.002 GFLOPs, 0.001% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(0.262 M, 0.695% Params, 1.049 GFLOPs, 0.438% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.002 M, 0.005% Params, 0.008 GFLOPs, 0.003% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.006 GFLOPs, 0.003% FLOPs, inplace=True)
      )
    )
    (layer4): ResLayer(
      14.965 M, 39.649% Params, 16.551 GFLOPs, 6.916% FLOPs, 
      (0): Bottleneck(
        6.04 M, 16.002% Params, 7.62 GFLOPs, 3.184% FLOPs, 
        (conv1): Conv2d(0.524 M, 1.389% Params, 2.097 GFLOPs, 0.876% FLOPs, 1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.004 GFLOPs, 0.002% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(2.359 M, 6.251% Params, 2.359 GFLOPs, 0.986% FLOPs, 512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.001 M, 0.003% Params, 0.001 GFLOPs, 0.000% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(1.049 M, 2.778% Params, 1.049 GFLOPs, 0.438% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.004 M, 0.011% Params, 0.004 GFLOPs, 0.002% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.002% FLOPs, inplace=True)
        (downsample): Sequential(
          2.101 M, 5.567% Params, 2.101 GFLOPs, 0.878% FLOPs, 
          (0): Conv2d(2.097 M, 5.556% Params, 2.097 GFLOPs, 0.876% FLOPs, 1024, 2048, kernel_size=(1, 1), stride=(2, 2), bias=False)
          (1): BatchNorm2d(0.004 M, 0.011% Params, 0.004 GFLOPs, 0.002% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
      )
      (1): Bottleneck(
        4.463 M, 11.824% Params, 4.466 GFLOPs, 1.866% FLOPs, 
        (conv1): Conv2d(1.049 M, 2.778% Params, 1.049 GFLOPs, 0.438% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.001 GFLOPs, 0.000% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(2.359 M, 6.251% Params, 2.359 GFLOPs, 0.986% FLOPs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.001 M, 0.003% Params, 0.001 GFLOPs, 0.000% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(1.049 M, 2.778% Params, 1.049 GFLOPs, 0.438% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.004 M, 0.011% Params, 0.004 GFLOPs, 0.002% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.003 GFLOPs, 0.001% FLOPs, inplace=True)
      )
      (2): Bottleneck(
        4.463 M, 11.824% Params, 4.466 GFLOPs, 1.866% FLOPs, 
        (conv1): Conv2d(1.049 M, 2.778% Params, 1.049 GFLOPs, 0.438% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn1): BatchNorm2d(0.001 M, 0.003% Params, 0.001 GFLOPs, 0.000% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv2): Conv2d(2.359 M, 6.251% Params, 2.359 GFLOPs, 0.986% FLOPs, 512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(0.001 M, 0.003% Params, 0.001 GFLOPs, 0.000% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (conv3): Conv2d(1.049 M, 2.778% Params, 1.049 GFLOPs, 0.438% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn3): BatchNorm2d(0.004 M, 0.011% Params, 0.004 GFLOPs, 0.002% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(0.0 M, 0.000% Params, 0.003 GFLOPs, 0.001% FLOPs, inplace=True)
      )
    )
  )
  init_cfg={'type': 'Pretrained', 'checkpoint': 'torchvision://resnet50'}
  (neck): FPN(
    7.997 M, 21.189% Params, 17.335 GFLOPs, 7.244% FLOPs, 
    (lateral_convs): ModuleList(
      0.918 M, 2.433% Params, 3.675 GFLOPs, 1.536% FLOPs, 
      (0): ConvModule(
        0.131 M, 0.348% Params, 2.101 GFLOPs, 0.878% FLOPs, 
        (conv): Conv2d(0.131 M, 0.348% Params, 2.101 GFLOPs, 0.878% FLOPs, 512, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ConvModule(
        0.262 M, 0.695% Params, 1.05 GFLOPs, 0.439% FLOPs, 
        (conv): Conv2d(0.262 M, 0.695% Params, 1.05 GFLOPs, 0.439% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ConvModule(
        0.525 M, 1.390% Params, 0.525 GFLOPs, 0.219% FLOPs, 
        (conv): Conv2d(0.525 M, 1.390% Params, 0.525 GFLOPs, 0.219% FLOPs, 2048, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (fpn_convs): ModuleList(
      7.079 M, 18.756% Params, 13.66 GFLOPs, 5.708% FLOPs, 
      (0): ConvModule(
        0.59 M, 1.563% Params, 9.441 GFLOPs, 3.945% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 9.441 GFLOPs, 3.945% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (1): ConvModule(
        0.59 M, 1.563% Params, 2.36 GFLOPs, 0.986% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 2.36 GFLOPs, 0.986% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (2): ConvModule(
        0.59 M, 1.563% Params, 0.59 GFLOPs, 0.247% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 0.59 GFLOPs, 0.247% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (3): ConvModule(
        4.719 M, 12.502% Params, 1.227 GFLOPs, 0.513% FLOPs, 
        (conv): Conv2d(4.719 M, 12.502% Params, 1.227 GFLOPs, 0.513% FLOPs, 2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
      (4): ConvModule(
        0.59 M, 1.563% Params, 0.041 GFLOPs, 0.017% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 0.041 GFLOPs, 0.017% FLOPs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
  )
  init_cfg={'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
  (bbox_head): RetinaHead(
    6.463 M, 17.124% Params, 137.904 GFLOPs, 57.624% FLOPs, 
    (loss_cls): FocalLoss(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
    (loss_bbox): L1Loss(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
    (relu): ReLU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, inplace=True)
    (cls_convs): ModuleList(
      2.36 M, 6.254% Params, 50.367 GFLOPs, 21.046% FLOPs, 
      (0): ConvModule(
        0.59 M, 1.563% Params, 12.592 GFLOPs, 5.262% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 12.586 GFLOPs, 5.259% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.002% FLOPs, inplace=True)
      )
      (1): ConvModule(
        0.59 M, 1.563% Params, 12.592 GFLOPs, 5.262% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 12.586 GFLOPs, 5.259% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.002% FLOPs, inplace=True)
      )
      (2): ConvModule(
        0.59 M, 1.563% Params, 12.592 GFLOPs, 5.262% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 12.586 GFLOPs, 5.259% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.002% FLOPs, inplace=True)
      )
      (3): ConvModule(
        0.59 M, 1.563% Params, 12.592 GFLOPs, 5.262% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 12.586 GFLOPs, 5.259% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.002% FLOPs, inplace=True)
      )
    )
    (reg_convs): ModuleList(
      2.36 M, 6.254% Params, 50.367 GFLOPs, 21.046% FLOPs, 
      (0): ConvModule(
        0.59 M, 1.563% Params, 12.592 GFLOPs, 5.262% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 12.586 GFLOPs, 5.259% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.002% FLOPs, inplace=True)
      )
      (1): ConvModule(
        0.59 M, 1.563% Params, 12.592 GFLOPs, 5.262% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 12.586 GFLOPs, 5.259% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.002% FLOPs, inplace=True)
      )
      (2): ConvModule(
        0.59 M, 1.563% Params, 12.592 GFLOPs, 5.262% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 12.586 GFLOPs, 5.259% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.002% FLOPs, inplace=True)
      )
      (3): ConvModule(
        0.59 M, 1.563% Params, 12.592 GFLOPs, 5.262% FLOPs, 
        (conv): Conv2d(0.59 M, 1.563% Params, 12.586 GFLOPs, 5.259% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.002% FLOPs, inplace=True)
      )
    )
    (retina_cls): Conv2d(1.66 M, 4.397% Params, 35.399 GFLOPs, 14.792% FLOPs, 256, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (retina_reg): Conv2d(0.083 M, 0.220% Params, 1.77 GFLOPs, 0.740% FLOPs, 256, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  init_cfg={'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01, 'override': {'type': 'Normal', 'name': 'retina_cls', 'std': 0.01, 'bias_prob': 0.01}}
)
==============================
Input shape: (3, 1280, 800)
Flops: 239.32 GFLOPs
Params: 37.74 M

</code></pre>
</details>

<summary>2. RetinaNet_RepLKNet13_fpn_1x_minicoco</summary>
<details>
<pre><code>

RetinaNet(
  301.72 M, 100.000% Params, 1234.986 GFLOPs, 100.000% FLOPs, 
  (backbone): RepLKNet(
    287.259 M, 95.207% Params, 1079.747 GFLOPs, 87.430% FLOPs, 
    (stem): ModuleList(
      0.079 M, 0.026% Params, 19.923 GFLOPs, 1.613% FLOPs, 
      (0): Sequential(
        0.007 M, 0.002% Params, 1.966 GFLOPs, 0.159% FLOPs, 
        (conv): Conv2d(0.007 M, 0.002% Params, 1.769 GFLOPs, 0.143% FLOPs, 3, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.131 GFLOPs, 0.011% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlinear): ReLU(0.0 M, 0.000% Params, 0.066 GFLOPs, 0.005% FLOPs, )
      )
      (1): Sequential(
        0.003 M, 0.001% Params, 0.786 GFLOPs, 0.064% FLOPs, 
        (conv): Conv2d(0.002 M, 0.001% Params, 0.59 GFLOPs, 0.048% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.131 GFLOPs, 0.011% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlinear): ReLU(0.0 M, 0.000% Params, 0.066 GFLOPs, 0.005% FLOPs, )
      )
      (2): Sequential(
        0.066 M, 0.022% Params, 16.974 GFLOPs, 1.374% FLOPs, 
        (conv): Conv2d(0.066 M, 0.022% Params, 16.777 GFLOPs, 1.358% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.131 GFLOPs, 0.011% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlinear): ReLU(0.0 M, 0.000% Params, 0.066 GFLOPs, 0.005% FLOPs, )
      )
      (3): Sequential(
        0.003 M, 0.001% Params, 0.197 GFLOPs, 0.016% FLOPs, 
        (conv): Conv2d(0.002 M, 0.001% Params, 0.147 GFLOPs, 0.012% FLOPs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
        (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlinear): ReLU(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, )
      )
    )
    (stages): ModuleList(
      284.381 M, 94.253% Params, 1034.314 GFLOPs, 83.751% FLOPs, 
      (0): RepLKNetStage(
        1.421 M, 0.471% Params, 91.03 GFLOPs, 7.371% FLOPs, 
        (blocks): ModuleList(
          1.421 M, 0.471% Params, 91.03 GFLOPs, 7.371% FLOPs, 
          (0): RepLKBlock(
            0.183 M, 0.061% Params, 11.764 GFLOPs, 0.953% FLOPs, 
            (pw1): Sequential(
              0.066 M, 0.022% Params, 4.243 GFLOPs, 0.344% FLOPs, 
              (conv): Conv2d(0.066 M, 0.022% Params, 4.194 GFLOPs, 0.340% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, )
            )
            (pw2): Sequential(
              0.066 M, 0.022% Params, 4.227 GFLOPs, 0.342% FLOPs, 
              (conv): Conv2d(0.066 M, 0.022% Params, 4.194 GFLOPs, 0.340% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.051 M, 0.017% Params, 3.244 GFLOPs, 0.263% FLOPs, 
              (lkb_origin): Sequential(
                0.044 M, 0.015% Params, 2.802 GFLOPs, 0.227% FLOPs, 
                (conv): Conv2d(0.043 M, 0.014% Params, 2.769 GFLOPs, 0.224% FLOPs, 256, 256, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=256, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.007 M, 0.002% Params, 0.442 GFLOPs, 0.036% FLOPs, 
                (conv): Conv2d(0.006 M, 0.002% Params, 0.41 GFLOPs, 0.033% FLOPs, 256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (1): ConvFFN(
            0.527 M, 0.175% Params, 33.751 GFLOPs, 2.733% FLOPs, 
            (drop_path): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              0.264 M, 0.088% Params, 16.908 GFLOPs, 1.369% FLOPs, 
              (conv): Conv2d(0.262 M, 0.087% Params, 16.777 GFLOPs, 1.358% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.131 GFLOPs, 0.011% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              0.263 M, 0.087% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(0.262 M, 0.087% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (2): RepLKBlock(
            0.183 M, 0.061% Params, 11.764 GFLOPs, 0.953% FLOPs, 
            (pw1): Sequential(
              0.066 M, 0.022% Params, 4.243 GFLOPs, 0.344% FLOPs, 
              (conv): Conv2d(0.066 M, 0.022% Params, 4.194 GFLOPs, 0.340% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, )
            )
            (pw2): Sequential(
              0.066 M, 0.022% Params, 4.227 GFLOPs, 0.342% FLOPs, 
              (conv): Conv2d(0.066 M, 0.022% Params, 4.194 GFLOPs, 0.340% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.051 M, 0.017% Params, 3.244 GFLOPs, 0.263% FLOPs, 
              (lkb_origin): Sequential(
                0.044 M, 0.015% Params, 2.802 GFLOPs, 0.227% FLOPs, 
                (conv): Conv2d(0.043 M, 0.014% Params, 2.769 GFLOPs, 0.224% FLOPs, 256, 256, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=256, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.007 M, 0.002% Params, 0.442 GFLOPs, 0.036% FLOPs, 
                (conv): Conv2d(0.006 M, 0.002% Params, 0.41 GFLOPs, 0.033% FLOPs, 256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (3): ConvFFN(
            0.527 M, 0.175% Params, 33.751 GFLOPs, 2.733% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              0.264 M, 0.088% Params, 16.908 GFLOPs, 1.369% FLOPs, 
              (conv): Conv2d(0.262 M, 0.087% Params, 16.777 GFLOPs, 1.358% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.131 GFLOPs, 0.011% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              0.263 M, 0.087% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(0.262 M, 0.087% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
        (norm): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      )
      (1): RepLKNetStage(
        5.464 M, 1.811% Params, 87.458 GFLOPs, 7.082% FLOPs, 
        (blocks): ModuleList(
          5.464 M, 1.811% Params, 87.458 GFLOPs, 7.082% FLOPs, 
          (0): RepLKBlock(
            0.629 M, 0.208% Params, 10.076 GFLOPs, 0.816% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.087% Params, 4.219 GFLOPs, 0.342% FLOPs, 
              (conv): Conv2d(0.262 M, 0.087% Params, 4.194 GFLOPs, 0.340% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.001% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.087% Params, 4.211 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(0.262 M, 0.087% Params, 4.194 GFLOPs, 0.340% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.101 M, 0.034% Params, 1.622 GFLOPs, 0.131% FLOPs, 
              (lkb_origin): Sequential(
                0.088 M, 0.029% Params, 1.401 GFLOPs, 0.113% FLOPs, 
                (conv): Conv2d(0.087 M, 0.029% Params, 1.384 GFLOPs, 0.112% FLOPs, 512, 512, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.005% Params, 0.221 GFLOPs, 0.018% FLOPs, 
                (conv): Conv2d(0.013 M, 0.004% Params, 0.205 GFLOPs, 0.017% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.001% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (1): ConvFFN(
            2.103 M, 0.697% Params, 33.653 GFLOPs, 2.725% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 0.349% Params, 16.843 GFLOPs, 1.364% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 16.777 GFLOPs, 1.358% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.066 GFLOPs, 0.005% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 0.348% Params, 16.794 GFLOPs, 1.360% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 16.777 GFLOPs, 1.358% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (2): RepLKBlock(
            0.629 M, 0.208% Params, 10.076 GFLOPs, 0.816% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.087% Params, 4.219 GFLOPs, 0.342% FLOPs, 
              (conv): Conv2d(0.262 M, 0.087% Params, 4.194 GFLOPs, 0.340% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.001% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.087% Params, 4.211 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(0.262 M, 0.087% Params, 4.194 GFLOPs, 0.340% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.101 M, 0.034% Params, 1.622 GFLOPs, 0.131% FLOPs, 
              (lkb_origin): Sequential(
                0.088 M, 0.029% Params, 1.401 GFLOPs, 0.113% FLOPs, 
                (conv): Conv2d(0.087 M, 0.029% Params, 1.384 GFLOPs, 0.112% FLOPs, 512, 512, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.005% Params, 0.221 GFLOPs, 0.018% FLOPs, 
                (conv): Conv2d(0.013 M, 0.004% Params, 0.205 GFLOPs, 0.017% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.001% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (3): ConvFFN(
            2.103 M, 0.697% Params, 33.653 GFLOPs, 2.725% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 0.349% Params, 16.843 GFLOPs, 1.364% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 16.777 GFLOPs, 1.358% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.066 GFLOPs, 0.005% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 0.348% Params, 16.794 GFLOPs, 1.360% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 16.777 GFLOPs, 1.358% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
        (norm): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      )
      (2): RepLKNetStage(
        192.725 M, 63.875% Params, 771.047 GFLOPs, 62.434% FLOPs, 
        (blocks): ModuleList(
          192.725 M, 63.875% Params, 771.047 GFLOPs, 62.434% FLOPs, 
          (0): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (1): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (2): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (3): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (4): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (5): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (6): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (7): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (8): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (9): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (10): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (11): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (12): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (13): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (14): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (15): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (16): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (17): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (18): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (19): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (20): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (21): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (22): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (23): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (24): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (25): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (26): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (27): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (28): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (29): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (30): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (31): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (32): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (33): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (34): RepLKBlock(
            2.306 M, 0.764% Params, 9.232 GFLOPs, 0.748% FLOPs, 
            (pw1): Sequential(
              1.051 M, 0.348% Params, 4.207 GFLOPs, 0.341% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 0.348% Params, 4.202 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(1.049 M, 0.348% Params, 4.194 GFLOPs, 0.340% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.067% Params, 0.811 GFLOPs, 0.066% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.058% Params, 0.7 GFLOPs, 0.057% FLOPs, 
                (conv): Conv2d(0.173 M, 0.057% Params, 0.692 GFLOPs, 0.056% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.009% Params, 0.111 GFLOPs, 0.009% FLOPs, 
                (conv): Conv2d(0.026 M, 0.008% Params, 0.102 GFLOPs, 0.008% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (35): ConvFFN(
            8.401 M, 2.784% Params, 33.604 GFLOPs, 2.721% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 1.393% Params, 16.81 GFLOPs, 1.361% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.003% Params, 0.033 GFLOPs, 0.003% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 1.391% Params, 16.785 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 16.777 GFLOPs, 1.358% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
        (norm): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      )
      (3): RepLKNetStage(
        84.771 M, 28.096% Params, 84.779 GFLOPs, 6.865% FLOPs, 
        (blocks): ModuleList(
          84.771 M, 28.096% Params, 84.779 GFLOPs, 6.865% FLOPs, 
          (0): RepLKBlock(
            8.806 M, 2.919% Params, 8.81 GFLOPs, 0.713% FLOPs, 
            (pw1): Sequential(
              4.198 M, 1.391% Params, 4.2 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 4.194 GFLOPs, 0.340% FLOPs, 2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              4.198 M, 1.391% Params, 4.198 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 4.194 GFLOPs, 0.340% FLOPs, 2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.406 M, 0.134% Params, 0.406 GFLOPs, 0.033% FLOPs, 
              (lkb_origin): Sequential(
                0.35 M, 0.116% Params, 0.35 GFLOPs, 0.028% FLOPs, 
                (conv): Conv2d(0.346 M, 0.115% Params, 0.346 GFLOPs, 0.028% FLOPs, 2048, 2048, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=2048, bias=False)
                (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.055 M, 0.018% Params, 0.055 GFLOPs, 0.004% FLOPs, 
                (conv): Conv2d(0.051 M, 0.017% Params, 0.051 GFLOPs, 0.004% FLOPs, 2048, 2048, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2048, bias=False)
                (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (1): ConvFFN(
            33.579 M, 11.129% Params, 33.579 GFLOPs, 2.719% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              16.794 M, 5.566% Params, 16.794 GFLOPs, 1.360% FLOPs, 
              (conv): Conv2d(16.777 M, 5.561% Params, 16.777 GFLOPs, 1.358% FLOPs, 2048, 8192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.016 M, 0.005% Params, 0.016 GFLOPs, 0.001% FLOPs, 8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              16.781 M, 5.562% Params, 16.781 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(16.777 M, 5.561% Params, 16.777 GFLOPs, 1.358% FLOPs, 8192, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (2): RepLKBlock(
            8.806 M, 2.919% Params, 8.81 GFLOPs, 0.713% FLOPs, 
            (pw1): Sequential(
              4.198 M, 1.391% Params, 4.2 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 4.194 GFLOPs, 0.340% FLOPs, 2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              4.198 M, 1.391% Params, 4.198 GFLOPs, 0.340% FLOPs, 
              (conv): Conv2d(4.194 M, 1.390% Params, 4.194 GFLOPs, 0.340% FLOPs, 2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.406 M, 0.134% Params, 0.406 GFLOPs, 0.033% FLOPs, 
              (lkb_origin): Sequential(
                0.35 M, 0.116% Params, 0.35 GFLOPs, 0.028% FLOPs, 
                (conv): Conv2d(0.346 M, 0.115% Params, 0.346 GFLOPs, 0.028% FLOPs, 2048, 2048, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=2048, bias=False)
                (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.055 M, 0.018% Params, 0.055 GFLOPs, 0.004% FLOPs, 
                (conv): Conv2d(0.051 M, 0.017% Params, 0.051 GFLOPs, 0.004% FLOPs, 2048, 2048, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=2048, bias=False)
                (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (3): ConvFFN(
            33.579 M, 11.129% Params, 33.579 GFLOPs, 2.719% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              16.794 M, 5.566% Params, 16.794 GFLOPs, 1.360% FLOPs, 
              (conv): Conv2d(16.777 M, 5.561% Params, 16.777 GFLOPs, 1.358% FLOPs, 2048, 8192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.016 M, 0.005% Params, 0.016 GFLOPs, 0.001% FLOPs, 8192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              16.781 M, 5.562% Params, 16.781 GFLOPs, 1.359% FLOPs, 
              (conv): Conv2d(16.777 M, 5.561% Params, 16.777 GFLOPs, 1.358% FLOPs, 8192, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
        (norm): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      )
    )
    (transitions): ModuleList(
      2.799 M, 0.928% Params, 25.51 GFLOPs, 2.066% FLOPs, 
      (0): Sequential(
        0.138 M, 0.046% Params, 8.585 GFLOPs, 0.695% FLOPs, 
        (0): Sequential(
          0.132 M, 0.044% Params, 8.487 GFLOPs, 0.687% FLOPs, 
          (conv): Conv2d(0.131 M, 0.043% Params, 8.389 GFLOPs, 0.679% FLOPs, 256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.066 GFLOPs, 0.005% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.033 GFLOPs, 0.003% FLOPs, )
        )
        (1): Sequential(
          0.006 M, 0.002% Params, 0.098 GFLOPs, 0.008% FLOPs, 
          (conv): Conv2d(0.005 M, 0.002% Params, 0.074 GFLOPs, 0.006% FLOPs, 512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, bias=False)
          (bn): BatchNorm2d(0.001 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.001% FLOPs, )
        )
      )
      (1): Sequential(
        0.538 M, 0.178% Params, 8.487 GFLOPs, 0.687% FLOPs, 
        (0): Sequential(
          0.526 M, 0.174% Params, 8.438 GFLOPs, 0.683% FLOPs, 
          (conv): Conv2d(0.524 M, 0.174% Params, 8.389 GFLOPs, 0.679% FLOPs, 512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.033 GFLOPs, 0.003% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.001% FLOPs, )
        )
        (1): Sequential(
          0.011 M, 0.004% Params, 0.049 GFLOPs, 0.004% FLOPs, 
          (conv): Conv2d(0.009 M, 0.003% Params, 0.037 GFLOPs, 0.003% FLOPs, 1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1024, bias=False)
          (bn): BatchNorm2d(0.002 M, 0.001% Params, 0.008 GFLOPs, 0.001% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.000% FLOPs, )
        )
      )
      (2): Sequential(
        2.124 M, 0.704% Params, 8.438 GFLOPs, 0.683% FLOPs, 
        (0): Sequential(
          2.101 M, 0.696% Params, 8.413 GFLOPs, 0.681% FLOPs, 
          (conv): Conv2d(2.097 M, 0.695% Params, 8.389 GFLOPs, 0.679% FLOPs, 1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.016 GFLOPs, 0.001% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.001% FLOPs, )
        )
        (1): Sequential(
          0.023 M, 0.007% Params, 0.025 GFLOPs, 0.002% FLOPs, 
          (conv): Conv2d(0.018 M, 0.006% Params, 0.018 GFLOPs, 0.001% FLOPs, 2048, 2048, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=2048, bias=False)
          (bn): BatchNorm2d(0.004 M, 0.001% Params, 0.004 GFLOPs, 0.000% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
        )
      )
    )
  )
  (neck): FPN(
    7.997 M, 2.651% Params, 17.335 GFLOPs, 1.404% FLOPs, 
    (lateral_convs): ModuleList(
      0.918 M, 0.304% Params, 3.675 GFLOPs, 0.298% FLOPs, 
      (0): ConvModule(
        0.131 M, 0.044% Params, 2.101 GFLOPs, 0.170% FLOPs, 
        (conv): Conv2d(0.131 M, 0.044% Params, 2.101 GFLOPs, 0.170% FLOPs, 512, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ConvModule(
        0.262 M, 0.087% Params, 1.05 GFLOPs, 0.085% FLOPs, 
        (conv): Conv2d(0.262 M, 0.087% Params, 1.05 GFLOPs, 0.085% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ConvModule(
        0.525 M, 0.174% Params, 0.525 GFLOPs, 0.042% FLOPs, 
        (conv): Conv2d(0.525 M, 0.174% Params, 0.525 GFLOPs, 0.042% FLOPs, 2048, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (fpn_convs): ModuleList(
      7.079 M, 2.346% Params, 13.66 GFLOPs, 1.106% FLOPs, 
      (0): ConvModule(
        0.59 M, 0.196% Params, 9.441 GFLOPs, 0.764% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 9.441 GFLOPs, 0.764% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (1): ConvModule(
        0.59 M, 0.196% Params, 2.36 GFLOPs, 0.191% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 2.36 GFLOPs, 0.191% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (2): ConvModule(
        0.59 M, 0.196% Params, 0.59 GFLOPs, 0.048% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 0.59 GFLOPs, 0.048% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (3): ConvModule(
        4.719 M, 1.564% Params, 1.227 GFLOPs, 0.099% FLOPs, 
        (conv): Conv2d(4.719 M, 1.564% Params, 1.227 GFLOPs, 0.099% FLOPs, 2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
      (4): ConvModule(
        0.59 M, 0.196% Params, 0.041 GFLOPs, 0.003% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 0.041 GFLOPs, 0.003% FLOPs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
  )
  init_cfg={'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
  (bbox_head): RetinaHead(
    6.463 M, 2.142% Params, 137.904 GFLOPs, 11.166% FLOPs, 
    (loss_cls): FocalLoss(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
    (loss_bbox): L1Loss(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
    (relu): ReLU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, inplace=True)
    (cls_convs): ModuleList(
      2.36 M, 0.782% Params, 50.367 GFLOPs, 4.078% FLOPs, 
      (0): ConvModule(
        0.59 M, 0.196% Params, 12.592 GFLOPs, 1.020% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 12.586 GFLOPs, 1.019% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.000% FLOPs, inplace=True)
      )
      (1): ConvModule(
        0.59 M, 0.196% Params, 12.592 GFLOPs, 1.020% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 12.586 GFLOPs, 1.019% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.000% FLOPs, inplace=True)
      )
      (2): ConvModule(
        0.59 M, 0.196% Params, 12.592 GFLOPs, 1.020% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 12.586 GFLOPs, 1.019% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.000% FLOPs, inplace=True)
      )
      (3): ConvModule(
        0.59 M, 0.196% Params, 12.592 GFLOPs, 1.020% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 12.586 GFLOPs, 1.019% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.000% FLOPs, inplace=True)
      )
    )
    (reg_convs): ModuleList(
      2.36 M, 0.782% Params, 50.367 GFLOPs, 4.078% FLOPs, 
      (0): ConvModule(
        0.59 M, 0.196% Params, 12.592 GFLOPs, 1.020% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 12.586 GFLOPs, 1.019% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.000% FLOPs, inplace=True)
      )
      (1): ConvModule(
        0.59 M, 0.196% Params, 12.592 GFLOPs, 1.020% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 12.586 GFLOPs, 1.019% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.000% FLOPs, inplace=True)
      )
      (2): ConvModule(
        0.59 M, 0.196% Params, 12.592 GFLOPs, 1.020% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 12.586 GFLOPs, 1.019% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.000% FLOPs, inplace=True)
      )
      (3): ConvModule(
        0.59 M, 0.196% Params, 12.592 GFLOPs, 1.020% FLOPs, 
        (conv): Conv2d(0.59 M, 0.196% Params, 12.586 GFLOPs, 1.019% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.000% FLOPs, inplace=True)
      )
    )
    (retina_cls): Conv2d(1.66 M, 0.550% Params, 35.399 GFLOPs, 2.866% FLOPs, 256, 720, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (retina_reg): Conv2d(0.083 M, 0.028% Params, 1.77 GFLOPs, 0.143% FLOPs, 256, 36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  )
  init_cfg={'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01, 'override': {'type': 'Normal', 'name': 'retina_cls', 'std': 0.01, 'bias_prob': 0.01}}
)
==============================
Input shape: (3, 1280, 800)
Flops: 1234.99 GFLOPs
Params: 301.72 M
==============================

</code></pre>
</details>

<summary>3. RepLKNet-31B_1Kpretrain_fcos_1x_coco</summary>
<details>
<pre><code>


FCOS(
  87.166 M, 100.000% Params, 436.239 GFLOPs, 100.000% FLOPs, 
  (backbone): RepLKNet(
    78.837 M, 90.445% Params, 316.854 GFLOPs, 72.633% FLOPs, 
    (stem): ModuleList(
      0.023 M, 0.027% Params, 5.767 GFLOPs, 1.322% FLOPs, 
      (0): Sequential(
        0.004 M, 0.004% Params, 0.983 GFLOPs, 0.225% FLOPs, 
        (conv): Conv2d(0.003 M, 0.004% Params, 0.885 GFLOPs, 0.203% FLOPs, 3, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.066 GFLOPs, 0.015% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlinear): ReLU(0.0 M, 0.000% Params, 0.033 GFLOPs, 0.008% FLOPs, )
      )
      (1): Sequential(
        0.001 M, 0.002% Params, 0.393 GFLOPs, 0.090% FLOPs, 
        (conv): Conv2d(0.001 M, 0.001% Params, 0.295 GFLOPs, 0.068% FLOPs, 128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.066 GFLOPs, 0.015% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlinear): ReLU(0.0 M, 0.000% Params, 0.033 GFLOPs, 0.008% FLOPs, )
      )
      (2): Sequential(
        0.017 M, 0.019% Params, 4.293 GFLOPs, 0.984% FLOPs, 
        (conv): Conv2d(0.016 M, 0.019% Params, 4.194 GFLOPs, 0.961% FLOPs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.066 GFLOPs, 0.015% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlinear): ReLU(0.0 M, 0.000% Params, 0.033 GFLOPs, 0.008% FLOPs, )
      )
      (3): Sequential(
        0.001 M, 0.002% Params, 0.098 GFLOPs, 0.023% FLOPs, 
        (conv): Conv2d(0.001 M, 0.001% Params, 0.074 GFLOPs, 0.017% FLOPs, 128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=128, bias=False)
        (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.002% FLOPs, )
      )
    )
    (stages): ModuleList(
      78.103 M, 89.602% Params, 304.624 GFLOPs, 69.830% FLOPs, 
      (0): RepLKNetStage(
        0.586 M, 0.672% Params, 37.519 GFLOPs, 8.601% FLOPs, 
        (blocks): ModuleList(
          0.586 M, 0.672% Params, 37.519 GFLOPs, 8.601% FLOPs, 
          (0): RepLKBlock(
            0.16 M, 0.184% Params, 10.273 GFLOPs, 2.355% FLOPs, 
            (pw1): Sequential(
              0.017 M, 0.019% Params, 1.073 GFLOPs, 0.246% FLOPs, 
              (conv): Conv2d(0.016 M, 0.019% Params, 1.049 GFLOPs, 0.240% FLOPs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.002% FLOPs, )
            )
            (pw2): Sequential(
              0.017 M, 0.019% Params, 1.065 GFLOPs, 0.244% FLOPs, 
              (conv): Conv2d(0.016 M, 0.019% Params, 1.049 GFLOPs, 0.240% FLOPs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.127 M, 0.145% Params, 8.11 GFLOPs, 1.859% FLOPs, 
              (lkb_origin): Sequential(
                0.123 M, 0.141% Params, 7.889 GFLOPs, 1.808% FLOPs, 
                (conv): Conv2d(0.123 M, 0.141% Params, 7.873 GFLOPs, 1.805% FLOPs, 128, 128, kernel_size=(31, 31), stride=(1, 1), padding=(15, 15), groups=128, bias=False)
                (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.003 M, 0.004% Params, 0.221 GFLOPs, 0.051% FLOPs, 
                (conv): Conv2d(0.003 M, 0.004% Params, 0.205 GFLOPs, 0.047% FLOPs, 128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=False)
                (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.002% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (1): ConvFFN(
            0.133 M, 0.152% Params, 8.487 GFLOPs, 1.945% FLOPs, 
            (drop_path): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              0.067 M, 0.076% Params, 4.26 GFLOPs, 0.976% FLOPs, 
              (conv): Conv2d(0.066 M, 0.075% Params, 4.194 GFLOPs, 0.961% FLOPs, 128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.066 GFLOPs, 0.015% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              0.066 M, 0.075% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(0.066 M, 0.075% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (2): RepLKBlock(
            0.16 M, 0.184% Params, 10.273 GFLOPs, 2.355% FLOPs, 
            (pw1): Sequential(
              0.017 M, 0.019% Params, 1.073 GFLOPs, 0.246% FLOPs, 
              (conv): Conv2d(0.016 M, 0.019% Params, 1.049 GFLOPs, 0.240% FLOPs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.002% FLOPs, )
            )
            (pw2): Sequential(
              0.017 M, 0.019% Params, 1.065 GFLOPs, 0.244% FLOPs, 
              (conv): Conv2d(0.016 M, 0.019% Params, 1.049 GFLOPs, 0.240% FLOPs, 128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.127 M, 0.145% Params, 8.11 GFLOPs, 1.859% FLOPs, 
              (lkb_origin): Sequential(
                0.123 M, 0.141% Params, 7.889 GFLOPs, 1.808% FLOPs, 
                (conv): Conv2d(0.123 M, 0.141% Params, 7.873 GFLOPs, 1.805% FLOPs, 128, 128, kernel_size=(31, 31), stride=(1, 1), padding=(15, 15), groups=128, bias=False)
                (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.003 M, 0.004% Params, 0.221 GFLOPs, 0.051% FLOPs, 
                (conv): Conv2d(0.003 M, 0.004% Params, 0.205 GFLOPs, 0.047% FLOPs, 128, 128, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=128, bias=False)
                (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.002% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (3): ConvFFN(
            0.133 M, 0.152% Params, 8.487 GFLOPs, 1.945% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              0.067 M, 0.076% Params, 4.26 GFLOPs, 0.976% FLOPs, 
              (conv): Conv2d(0.066 M, 0.075% Params, 4.194 GFLOPs, 0.961% FLOPs, 128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.066 GFLOPs, 0.015% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              0.066 M, 0.075% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(0.066 M, 0.075% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, 128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
        (norm): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      )
      (1): RepLKNetStage(
        1.765 M, 2.025% Params, 28.262 GFLOPs, 6.479% FLOPs, 
        (blocks): ModuleList(
          1.765 M, 2.025% Params, 28.262 GFLOPs, 6.479% FLOPs, 
          (0): RepLKBlock(
            0.355 M, 0.408% Params, 5.693 GFLOPs, 1.305% FLOPs, 
            (pw1): Sequential(
              0.066 M, 0.076% Params, 1.061 GFLOPs, 0.243% FLOPs, 
              (conv): Conv2d(0.066 M, 0.075% Params, 1.049 GFLOPs, 0.240% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.001% FLOPs, )
            )
            (pw2): Sequential(
              0.066 M, 0.076% Params, 1.057 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.066 M, 0.075% Params, 1.049 GFLOPs, 0.240% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.223 M, 0.256% Params, 3.564 GFLOPs, 0.817% FLOPs, 
              (lkb_origin): Sequential(
                0.216 M, 0.248% Params, 3.453 GFLOPs, 0.792% FLOPs, 
                (conv): Conv2d(0.215 M, 0.247% Params, 3.445 GFLOPs, 0.790% FLOPs, 256, 256, kernel_size=(29, 29), stride=(1, 1), padding=(14, 14), groups=256, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.007 M, 0.008% Params, 0.111 GFLOPs, 0.025% FLOPs, 
                (conv): Conv2d(0.006 M, 0.007% Params, 0.102 GFLOPs, 0.023% FLOPs, 256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.001% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (1): ConvFFN(
            0.527 M, 0.605% Params, 8.438 GFLOPs, 1.934% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              0.264 M, 0.303% Params, 4.227 GFLOPs, 0.969% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 4.194 GFLOPs, 0.961% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.033 GFLOPs, 0.008% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              0.263 M, 0.301% Params, 4.202 GFLOPs, 0.963% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 4.194 GFLOPs, 0.961% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (2): RepLKBlock(
            0.355 M, 0.408% Params, 5.693 GFLOPs, 1.305% FLOPs, 
            (pw1): Sequential(
              0.066 M, 0.076% Params, 1.061 GFLOPs, 0.243% FLOPs, 
              (conv): Conv2d(0.066 M, 0.075% Params, 1.049 GFLOPs, 0.240% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.001% FLOPs, )
            )
            (pw2): Sequential(
              0.066 M, 0.076% Params, 1.057 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.066 M, 0.075% Params, 1.049 GFLOPs, 0.240% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.223 M, 0.256% Params, 3.564 GFLOPs, 0.817% FLOPs, 
              (lkb_origin): Sequential(
                0.216 M, 0.248% Params, 3.453 GFLOPs, 0.792% FLOPs, 
                (conv): Conv2d(0.215 M, 0.247% Params, 3.445 GFLOPs, 0.790% FLOPs, 256, 256, kernel_size=(29, 29), stride=(1, 1), padding=(14, 14), groups=256, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.007 M, 0.008% Params, 0.111 GFLOPs, 0.025% FLOPs, 
                (conv): Conv2d(0.006 M, 0.007% Params, 0.102 GFLOPs, 0.023% FLOPs, 256, 256, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=256, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.001% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (3): ConvFFN(
            0.527 M, 0.605% Params, 8.438 GFLOPs, 1.934% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              0.264 M, 0.303% Params, 4.227 GFLOPs, 0.969% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 4.194 GFLOPs, 0.961% FLOPs, 256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.033 GFLOPs, 0.008% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              0.263 M, 0.301% Params, 4.202 GFLOPs, 0.963% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 4.194 GFLOPs, 0.961% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
        (norm): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      )
      (2): RepLKNetStage(
        54.338 M, 62.338% Params, 217.424 GFLOPs, 49.841% FLOPs, 
        (blocks): ModuleList(
          54.338 M, 62.338% Params, 217.424 GFLOPs, 49.841% FLOPs, 
          (0): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (1): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (2): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (3): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (4): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (5): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (6): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (7): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (8): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (9): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (10): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (11): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (12): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (13): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (14): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (15): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (16): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (17): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (18): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (19): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (20): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (21): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (22): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (23): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (24): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (25): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (26): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (27): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (28): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (29): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (30): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (31): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (32): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (33): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (34): RepLKBlock(
            0.915 M, 1.050% Params, 3.666 GFLOPs, 0.840% FLOPs, 
            (pw1): Sequential(
              0.263 M, 0.302% Params, 1.055 GFLOPs, 0.242% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              0.263 M, 0.302% Params, 1.053 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(0.262 M, 0.301% Params, 1.049 GFLOPs, 0.240% FLOPs, 512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.388 M, 0.445% Params, 1.552 GFLOPs, 0.356% FLOPs, 
              (lkb_origin): Sequential(
                0.374 M, 0.429% Params, 1.497 GFLOPs, 0.343% FLOPs, 
                (conv): Conv2d(0.373 M, 0.428% Params, 1.493 GFLOPs, 0.342% FLOPs, 512, 512, kernel_size=(27, 27), stride=(1, 1), padding=(13, 13), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.014 M, 0.016% Params, 0.055 GFLOPs, 0.013% FLOPs, 
                (conv): Conv2d(0.013 M, 0.015% Params, 0.051 GFLOPs, 0.012% FLOPs, 512, 512, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=512, bias=False)
                (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (35): ConvFFN(
            2.103 M, 2.413% Params, 8.413 GFLOPs, 1.929% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              1.053 M, 1.208% Params, 4.211 GFLOPs, 0.965% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.004 M, 0.005% Params, 0.016 GFLOPs, 0.004% FLOPs, 2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              1.05 M, 1.204% Params, 4.198 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 4.194 GFLOPs, 0.961% FLOPs, 2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
        (norm): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      )
      (3): RepLKNetStage(
        21.414 M, 24.567% Params, 21.418 GFLOPs, 4.910% FLOPs, 
        (blocks): ModuleList(
          21.414 M, 24.567% Params, 21.418 GFLOPs, 4.910% FLOPs, 
          (0): RepLKBlock(
            2.306 M, 2.646% Params, 2.308 GFLOPs, 0.529% FLOPs, 
            (pw1): Sequential(
              1.051 M, 1.205% Params, 1.052 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 1.049 GFLOPs, 0.240% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.001 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 1.205% Params, 1.051 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 1.049 GFLOPs, 0.240% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.233% Params, 0.203 GFLOPs, 0.046% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.201% Params, 0.175 GFLOPs, 0.040% FLOPs, 
                (conv): Conv2d(0.173 M, 0.199% Params, 0.173 GFLOPs, 0.040% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.032% Params, 0.028 GFLOPs, 0.006% FLOPs, 
                (conv): Conv2d(0.026 M, 0.029% Params, 0.026 GFLOPs, 0.006% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.001 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (1): ConvFFN(
            8.401 M, 9.638% Params, 8.401 GFLOPs, 1.926% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 4.821% Params, 4.202 GFLOPs, 0.963% FLOPs, 
              (conv): Conv2d(4.194 M, 4.812% Params, 4.194 GFLOPs, 0.961% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.009% Params, 0.008 GFLOPs, 0.002% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 4.814% Params, 4.196 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(4.194 M, 4.812% Params, 4.194 GFLOPs, 0.961% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (2): RepLKBlock(
            2.306 M, 2.646% Params, 2.308 GFLOPs, 0.529% FLOPs, 
            (pw1): Sequential(
              1.051 M, 1.205% Params, 1.052 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 1.049 GFLOPs, 0.240% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (nonlinear): ReLU(0.0 M, 0.000% Params, 0.001 GFLOPs, 0.000% FLOPs, )
            )
            (pw2): Sequential(
              1.051 M, 1.205% Params, 1.051 GFLOPs, 0.241% FLOPs, 
              (conv): Conv2d(1.049 M, 1.203% Params, 1.049 GFLOPs, 0.240% FLOPs, 1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (large_kernel): ReparamLargeKernelConv(
              0.203 M, 0.233% Params, 0.203 GFLOPs, 0.046% FLOPs, 
              (lkb_origin): Sequential(
                0.175 M, 0.201% Params, 0.175 GFLOPs, 0.040% FLOPs, 
                (conv): Conv2d(0.173 M, 0.199% Params, 0.173 GFLOPs, 0.040% FLOPs, 1024, 1024, kernel_size=(13, 13), stride=(1, 1), padding=(6, 6), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
              (small_conv): Sequential(
                0.028 M, 0.032% Params, 0.028 GFLOPs, 0.006% FLOPs, 
                (conv): Conv2d(0.026 M, 0.029% Params, 0.026 GFLOPs, 0.006% FLOPs, 1024, 1024, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=1024, bias=False)
                (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              )
            )
            (lk_nonlinear): ReLU(0.0 M, 0.000% Params, 0.001 GFLOPs, 0.000% FLOPs, )
            (prelkb_bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
          (3): ConvFFN(
            8.401 M, 9.638% Params, 8.401 GFLOPs, 1.926% FLOPs, 
            (drop_path): DropPath(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
            (preffn_bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (pw1): Sequential(
              4.202 M, 4.821% Params, 4.202 GFLOPs, 0.963% FLOPs, 
              (conv): Conv2d(4.194 M, 4.812% Params, 4.194 GFLOPs, 0.961% FLOPs, 1024, 4096, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.008 M, 0.009% Params, 0.008 GFLOPs, 0.002% FLOPs, 4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (pw2): Sequential(
              4.196 M, 4.814% Params, 4.196 GFLOPs, 0.962% FLOPs, 
              (conv): Conv2d(4.194 M, 4.812% Params, 4.194 GFLOPs, 0.961% FLOPs, 4096, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
            (nonlinear): GELU(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
          )
        )
        (norm): Identity(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      )
    )
    (transitions): ModuleList(
      0.711 M, 0.816% Params, 6.463 GFLOPs, 1.482% FLOPs, 
      (0): Sequential(
        0.036 M, 0.041% Params, 2.195 GFLOPs, 0.503% FLOPs, 
        (0): Sequential(
          0.033 M, 0.038% Params, 2.146 GFLOPs, 0.492% FLOPs, 
          (conv): Conv2d(0.033 M, 0.038% Params, 2.097 GFLOPs, 0.481% FLOPs, 128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.033 GFLOPs, 0.008% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.016 GFLOPs, 0.004% FLOPs, )
        )
        (1): Sequential(
          0.003 M, 0.003% Params, 0.049 GFLOPs, 0.011% FLOPs, 
          (conv): Conv2d(0.002 M, 0.003% Params, 0.037 GFLOPs, 0.008% FLOPs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=256, bias=False)
          (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.008 GFLOPs, 0.002% FLOPs, 256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.001% FLOPs, )
        )
      )
      (1): Sequential(
        0.138 M, 0.158% Params, 2.146 GFLOPs, 0.492% FLOPs, 
        (0): Sequential(
          0.132 M, 0.152% Params, 2.122 GFLOPs, 0.486% FLOPs, 
          (conv): Conv2d(0.131 M, 0.150% Params, 2.097 GFLOPs, 0.481% FLOPs, 256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.016 GFLOPs, 0.004% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.008 GFLOPs, 0.002% FLOPs, )
        )
        (1): Sequential(
          0.006 M, 0.006% Params, 0.025 GFLOPs, 0.006% FLOPs, 
          (conv): Conv2d(0.005 M, 0.005% Params, 0.018 GFLOPs, 0.004% FLOPs, 512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=512, bias=False)
          (bn): BatchNorm2d(0.001 M, 0.001% Params, 0.004 GFLOPs, 0.001% FLOPs, 512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.002 GFLOPs, 0.000% FLOPs, )
        )
      )
      (2): Sequential(
        0.538 M, 0.617% Params, 2.122 GFLOPs, 0.486% FLOPs, 
        (0): Sequential(
          0.526 M, 0.604% Params, 2.109 GFLOPs, 0.484% FLOPs, 
          (conv): Conv2d(0.524 M, 0.601% Params, 2.097 GFLOPs, 0.481% FLOPs, 512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.008 GFLOPs, 0.002% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.004 GFLOPs, 0.001% FLOPs, )
        )
        (1): Sequential(
          0.011 M, 0.013% Params, 0.012 GFLOPs, 0.003% FLOPs, 
          (conv): Conv2d(0.009 M, 0.011% Params, 0.009 GFLOPs, 0.002% FLOPs, 1024, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=1024, bias=False)
          (bn): BatchNorm2d(0.002 M, 0.002% Params, 0.002 GFLOPs, 0.000% FLOPs, 1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (nonlinear): ReLU(0.0 M, 0.000% Params, 0.001 GFLOPs, 0.000% FLOPs, )
        )
      )
    )
  )
  (neck): FPN(
    3.41 M, 3.912% Params, 14.427 GFLOPs, 3.307% FLOPs, 
    (lateral_convs): ModuleList(
      0.46 M, 0.527% Params, 1.84 GFLOPs, 0.422% FLOPs, 
      (0): ConvModule(
        0.066 M, 0.075% Params, 1.053 GFLOPs, 0.241% FLOPs, 
        (conv): Conv2d(0.066 M, 0.075% Params, 1.053 GFLOPs, 0.241% FLOPs, 256, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (1): ConvModule(
        0.131 M, 0.151% Params, 0.525 GFLOPs, 0.120% FLOPs, 
        (conv): Conv2d(0.131 M, 0.151% Params, 0.525 GFLOPs, 0.120% FLOPs, 512, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (2): ConvModule(
        0.262 M, 0.301% Params, 0.262 GFLOPs, 0.060% FLOPs, 
        (conv): Conv2d(0.262 M, 0.301% Params, 0.262 GFLOPs, 0.060% FLOPs, 1024, 256, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    (fpn_convs): ModuleList(
      2.95 M, 3.385% Params, 12.586 GFLOPs, 2.885% FLOPs, 
      (0): ConvModule(
        0.59 M, 0.677% Params, 9.441 GFLOPs, 2.164% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 9.441 GFLOPs, 2.164% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (1): ConvModule(
        0.59 M, 0.677% Params, 2.36 GFLOPs, 0.541% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 2.36 GFLOPs, 0.541% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (2): ConvModule(
        0.59 M, 0.677% Params, 0.59 GFLOPs, 0.135% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 0.59 GFLOPs, 0.135% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (3): ConvModule(
        0.59 M, 0.677% Params, 0.153 GFLOPs, 0.035% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 0.153 GFLOPs, 0.035% FLOPs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
      (4): ConvModule(
        0.59 M, 0.677% Params, 0.041 GFLOPs, 0.009% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 0.041 GFLOPs, 0.009% FLOPs, 256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
  )
  init_cfg={'type': 'Xavier', 'layer': 'Conv2d', 'distribution': 'uniform'}
  (bbox_head): FCOSHead(
    4.919 M, 5.643% Params, 104.958 GFLOPs, 24.060% FLOPs, 
    (loss_cls): FocalLoss(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
    (loss_bbox): IoULoss(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
    (cls_convs): ModuleList(
      2.361 M, 2.709% Params, 50.389 GFLOPs, 11.551% FLOPs, 
      (0): ConvModule(
        0.59 M, 0.677% Params, 12.597 GFLOPs, 2.888% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 12.581 GFLOPs, 2.884% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (gn): GroupNorm(0.001 M, 0.001% Params, 0.011 GFLOPs, 0.003% FLOPs, 32, 256, eps=1e-05, affine=True)
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.001% FLOPs, inplace=True)
      )
      (1): ConvModule(
        0.59 M, 0.677% Params, 12.597 GFLOPs, 2.888% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 12.581 GFLOPs, 2.884% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (gn): GroupNorm(0.001 M, 0.001% Params, 0.011 GFLOPs, 0.003% FLOPs, 32, 256, eps=1e-05, affine=True)
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.001% FLOPs, inplace=True)
      )
      (2): ConvModule(
        0.59 M, 0.677% Params, 12.597 GFLOPs, 2.888% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 12.581 GFLOPs, 2.884% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (gn): GroupNorm(0.001 M, 0.001% Params, 0.011 GFLOPs, 0.003% FLOPs, 32, 256, eps=1e-05, affine=True)
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.001% FLOPs, inplace=True)
      )
      (3): ConvModule(
        0.59 M, 0.677% Params, 12.597 GFLOPs, 2.888% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 12.581 GFLOPs, 2.884% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (gn): GroupNorm(0.001 M, 0.001% Params, 0.011 GFLOPs, 0.003% FLOPs, 32, 256, eps=1e-05, affine=True)
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.001% FLOPs, inplace=True)
      )
    )
    (reg_convs): ModuleList(
      2.361 M, 2.709% Params, 50.389 GFLOPs, 11.551% FLOPs, 
      (0): ConvModule(
        0.59 M, 0.677% Params, 12.597 GFLOPs, 2.888% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 12.581 GFLOPs, 2.884% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (gn): GroupNorm(0.001 M, 0.001% Params, 0.011 GFLOPs, 0.003% FLOPs, 32, 256, eps=1e-05, affine=True)
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.001% FLOPs, inplace=True)
      )
      (1): ConvModule(
        0.59 M, 0.677% Params, 12.597 GFLOPs, 2.888% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 12.581 GFLOPs, 2.884% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (gn): GroupNorm(0.001 M, 0.001% Params, 0.011 GFLOPs, 0.003% FLOPs, 32, 256, eps=1e-05, affine=True)
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.001% FLOPs, inplace=True)
      )
      (2): ConvModule(
        0.59 M, 0.677% Params, 12.597 GFLOPs, 2.888% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 12.581 GFLOPs, 2.884% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (gn): GroupNorm(0.001 M, 0.001% Params, 0.011 GFLOPs, 0.003% FLOPs, 32, 256, eps=1e-05, affine=True)
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.001% FLOPs, inplace=True)
      )
      (3): ConvModule(
        0.59 M, 0.677% Params, 12.597 GFLOPs, 2.888% FLOPs, 
        (conv): Conv2d(0.59 M, 0.677% Params, 12.581 GFLOPs, 2.884% FLOPs, 256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (gn): GroupNorm(0.001 M, 0.001% Params, 0.011 GFLOPs, 0.003% FLOPs, 32, 256, eps=1e-05, affine=True)
        (activate): ReLU(0.0 M, 0.000% Params, 0.005 GFLOPs, 0.001% FLOPs, inplace=True)
      )
    )
    (conv_cls): Conv2d(0.184 M, 0.212% Params, 3.933 GFLOPs, 0.902% FLOPs, 256, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv_reg): Conv2d(0.009 M, 0.011% Params, 0.197 GFLOPs, 0.045% FLOPs, 256, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv_centerness): Conv2d(0.002 M, 0.003% Params, 0.049 GFLOPs, 0.011% FLOPs, 256, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (scales): ModuleList(
      0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, 
      (0): Scale(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      (1): Scale(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      (2): Scale(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      (3): Scale(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
      (4): Scale(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, )
    )
    (loss_centerness): CrossEntropyLoss(0.0 M, 0.000% Params, 0.0 GFLOPs, 0.000% FLOPs, avg_non_ignore=False)
  )
  init_cfg={'type': 'Normal', 'layer': 'Conv2d', 'std': 0.01, 'override': {'type': 'Normal', 'name': 'conv_cls', 'std': 0.01, 'bias_prob': 0.01}}
)
==============================
Input shape: (3, 1280, 800)
Flops: 436.24 GFLOPs
Params: 87.17 M
==============================
</code></pre>
</details>

