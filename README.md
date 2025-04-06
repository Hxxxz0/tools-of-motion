# toos-of-motion
some tools of motion
## smplx2humanml3d
- 根据 Humamml3d 要求下载必要库。
- 在 smplx 中下载 bodymodel 并保存到`./body_models/smplh/male/model.npz`文件夹（下载 smplh）。
- Download SMPL-X with removed head bun (NPZ, 392 MB) - Use this for SMPL-X Python codebase and AMASS data.

[手部动作重建 github](https://github.com/SeanChenxy/HandMesh)


joint2bvgh 在momask运行

另外两个jupyer 在 mdm


python 198*263 转换为 frames*22*3
```
import os
import torch
import numpy as np
from data_loaders.humanml.scripts.motion_process import recover_from_ric

# 定义目录路径
directory = '/home/user/dxc/motion/momask-codes-main/vis/mmm/'

# 遍历目录下的所有文件
for filename in os.listdir(directory):
    if filename.endswith('.npy'):
        file_path = os.path.join(directory, filename)
        try:
            # 加载 .npy 文件
            data = np.load(file_path)
            print(f"处理文件: {filename}, 数组的大小（形状）为: {data.shape}")

            # 进行处理
            joint1 = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()

            # 保存处理后的结果，替换原文件
            np.save(file_path, joint1)
            print(f"文件 {filename} 处理完成并替换。")

        except FileNotFoundError:
            print(f"文件 {filename} 不存在，请检查文件路径是否正确。")
        except Exception as e:
            print(f"处理文件 {filename} 时出现错误: {e}")
```
