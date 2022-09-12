[参考链接](https://stackoverflow.com/questions/69175642/droppath-in-timm-seems-like-a-dropout "参考链接")

#### 举例如下：
<details>
<summary>查看代码</summary>

<pre><code>
import torch
from torch.nn.functional import dropout

torch.manual_seed(2021)

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

x = torch.rand(3, 2, 2, 2)

# DropPath
d1_out = drop_path(x, drop_prob=0.33, training=True)

# Dropout
d2_out = dropout(x, p=0.33, training=True)
</code></pre>
</details>

<details>
<summary>查看输出</summary>

<pre><code>
# DropPath
print(d1_out)
#  tensor([[[[0.1947, 0.7662],
#            [1.1083, 1.0685]],
#           [[0.8515, 0.2467],
#            [0.0661, 1.4370]]],
#
#          [[[0.0000, 0.0000],
#            [0.0000, 0.0000]],
#           [[0.0000, 0.0000],
#            [0.0000, 0.0000]]],
#
#          [[[0.7658, 0.4417],
#            [1.1692, 1.1052]],
#           [[1.2014, 0.4532],
#            [1.4840, 0.7499]]]])

# Dropout
print(d2_out)
#  tensor([[[[0.1947, 0.7662],
#            [1.1083, 1.0685]],
#           [[0.8515, 0.2467],
#            [0.0661, 1.4370]]],
#
#          [[[0.0000, 0.1480],
#            [1.2083, 0.0000]],
#           [[1.2272, 0.1853],
#            [0.0000, 0.5385]]],
#
#          [[[0.7658, 0.0000],
#            [1.1692, 1.1052]],
#           [[1.2014, 0.4532],
#            [0.0000, 0.7499]]]])

</code></pre>
</details>

**对比两个输出，可以看出droppath是将样本从batch中整个丢弃，而dropout是在batch中每个样本里丢弃随机的元素**



