from motion_process import recover_from_ric


from plot_script import plot_3d_motion
import numpy as np
import paramUtil as paramUtil
import torch
import os
from scipy.fft import dct, idct
from tqdm import tqdm
import paramUtil as paramUtil
import matplotlib.pyplot as plt

import seaborn as sns

def recover_3d(pred_pose):
    pred_xyz = recover_from_ric(pred_pose.float(), 22)
    xyz = pred_xyz.reshape(1, -1, 22, 3)
    return xyz



def remove_low_f(motion,name,draw=False):
    sample=torch.from_numpy(motion)#(frame,263)
    vecs_list=[]
    for vec in sample.split(1,dim=1):
        vec=vec.squeeze().numpy()
        vec_f=dct(vec)
        vec_f[5:]=0
        vec_lf=idct(vec_f)

        vecs_list.append(vec_lf)
    motion_lf=np.array(vecs_list).transpose(1,0)
    # mean_f = np.mean(motion_lf,axis=1)
    # # vector_normalized = (mean_f - np.min(mean_f)) / (np.max(mean_f) - np.min(mean_f))
    # matrix = np.expand_dims(mean_f, axis=0)
    # matrix
    # 绘制热力图
    # plt.imshow(matrix, cmap='Reds', aspect='auto')
    # plt.colorbar()
    # plt.savefig(f'heatmap{name}.png', dpi=300, bbox_inches='tight')
    # plt.close()
    if draw==True:
        skeleton = paramUtil.t2m_kinematic_chain
        motion_joints_3d_lowf=recover_3d(torch.from_numpy(motion_lf)).numpy().squeeze()
        plot_3d_motion(f'/CV/xhr/xhr_2/trajectory--guidance-main/vis_results/{name}.mp4', skeleton, motion_joints_3d_lowf, dataset='humanml', title='demo', fps=20)
    return motion_lf
directory = '/CV/xhr/xhr_2/motion-diffusion-model/dataset/HumanML3D/new_joint_vecs'
path ='/CV/xhr/xhr_2/trajectory--guidance-main/motion_results/new_joint_vecs_lowf5'
files = os.listdir(directory)
i=0
for file in tqdm(files,desc="Coverting!"):
    if not os.path.exists(path):
    # 创建路径
        os.makedirs(path)

    motion =np.load(f'/CV/xhr/xhr_2/motion-diffusion-model/dataset/HumanML3D/new_joint_vecs/{file}',allow_pickle=True)
    skeleton = paramUtil.t2m_kinematic_chain
    motion_joints_3d_lowf=recover_3d(torch.from_numpy(motion)).numpy().squeeze()
    # plot_3d_motion(f'/CV/xhr/xhr_2/trajectory--guidance-main/vis_results/origin/{i}.mp4', skeleton, motion_joints_3d_lowf, dataset='humanml', title='demo', fps=20)
    # draw=True
    
    try:    
        motion_new=remove_low_f(motion,i)
        np.save(f'{path}/{file}',motion_new)
    except:
        print(file,motion.shape)        
        np.save(f'{path}/{file}',motion)
    
