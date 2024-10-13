import sys
import numpy as np
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
import pdb
import math


# 提供的关节映射信息
JOINT_MAP = {
    # 'BVH joint name': 'SMPLX joint index'
    'Hips': 0,
    'LeftUpLeg': 1,
    'RightUpLeg': 2,
    'Spine1': 3,
    'LeftLeg': 4,
    'RightLeg': 5,
    'Spine2': 6,
    'LeftFoot': 7,
    'RightFoot': 8,
    'Chest': 9,
    'LeftToe': 10,
    'RightToe': 11,
    'Neck': 12,
    'LeftShoulder': 13,
    'RightShoulder': 14,
    'Head': 15,
    'LeftArm': 16,
    'RightArm': 17,
    'LeftForeArm': 18,
    'RightForeArm': 19,
    'LeftHand': 20,
    'RightHand': 21,
    'FZLeftIndex1': 25,
    'FZLeftIndex2': 26,
    'FZLeftIndex3': 27,
    'FZLeftMiddle1': 28,
    'FZLeftMiddle2': 29,
    'FZLeftMiddle3': 30,
    'FZLeftPinky1': 31,
    'FZLeftPinky2': 32,
    'FZLeftPinky3': 33,
    'FZLeftRing1': 34,
    'FZLeftRing2': 35,
    'FZLeftRing3': 36,
    'FZLeftThumb1': 37,
    'FZLeftThumb2': 38,
    'FZLeftThumb3': 39,
    'FZRightIndex1': 40,
    'FZRightIndex2': 41,
    'FZRightIndex3': 42,
    'FZRightMiddle1': 43,
    'FZRightMiddle2': 44,
    'FZRightMiddle3': 45,
    'FZRightPinky1': 46,
    'FZRightPinky2': 47,
    'FZRightPinky3': 48,
    'FZRightRing1': 49,
    'FZRightRing2': 50,
    'FZRightRing3': 51,
    'FZRightThumb1': 52,
    'FZRightThumb2': 53,
    'FZRightThumb3': 54,
}


def bvh_to_smplx(bvh_file, n_frames=None):
    with open(bvh_file, 'r') as f:
        mocap = Bvh(f.read())

    if n_frames is None:
        num_frames = len(mocap.frames)
    else:
        num_frames = min(n_frames, len(mocap.frames))

    # 计算降采样后的帧数
    num_frames_downsampled = math.ceil(num_frames / 1)

    smplx_poses = np.zeros((num_frames_downsampled, 165))
    smplx_trans = np.zeros((num_frames_downsampled, 3))

    bvh_joint_names = set(mocap.get_joints_names())

    # 定义一个从Y轴负向到Z轴正向的旋转
    rotation_correction = R.from_euler('XYZ', [90, 0, 0], degrees=True)

    for i in range(0, num_frames, 1):
        print('Processing frame {}/{}'.format(i, num_frames), end='\r')
        for joint_name, joint_index in JOINT_MAP.items():
            # print(joint_name, joint_index)
            # 检查关节是否存在于BVH文件中
            # if joint_name not in bvh_joint_names:
            #     continue

            # 提取关节旋转
            rotation = R.from_euler('ZYX', mocap.frame_joint_channels(i, joint_name,  ['Zrotation', 'Yrotation', 'Xrotation']), degrees=True)

            # 仅对根关节（Hips）应用朝向校正
            if joint_name == 'Hips':
                # rotation = rotation * rotation_correction
                rotation = rotation_correction * rotation

            smplx_poses[i//1, 3 * joint_index:3 * (joint_index + 1)] = rotation.as_rotvec()

            # 提取根关节平移
            if joint_name == 'Hips':
                x, y, z = mocap.frame_joint_channels(i, joint_name, ['Xposition', 'Yposition', 'Zposition'])
                smplx_trans[i // 1] = np.array([x, -z, y])

                # smplx_trans[i] = mocap.frame_joint_channels(i, joint_name, ['Zposition', 'Yposition', 'Xposition'])

    # 应用朝向校正
    # smplx_trans = rotation_correction_trans.apply(smplx_trans)

    # 反转Y轴平移方向
    # smplx_trans[:, 1] *= -1

    # 应用整体缩放
    scale_factor = 0.009
    smplx_trans *= scale_factor

    return smplx_trans, smplx_poses


def save_npz(output_file, smplx_trans, smplx_poses, gender='neutral', model_type='smplx', frame_rate=30):
    np.savez(output_file, trans=smplx_trans, poses=smplx_poses, gender=gender, surface_model_type=model_type,
             mocap_framerate=frame_rate, betas=np.zeros(16))




if __name__ == '__main__':
    '''
    pip install bvh
    cd process
    python bvh2smplx.py 2_scott_0_85_85.bvh 2_scott_0_85_85.npz
    '''
    bvh_file = "G:\paper\capture\Data 2024-09-26 22-14-17.bvh_Skeleton.bvh"
    output_file = "2.npz"

    # import bvh
    # with open("10_kieks_0_96_96_1.bvh", "r") as f:
    #     mocap = bvh.Bvh(f.read())
    # joints = []
    # for joint in mocap.get_joints_names():
    #     joints.append(joint)
    # print(joints)

    smplx_trans, smplx_poses = bvh_to_smplx(bvh_file, n_frames=3000)

    with open(bvh_file, 'r') as f:
        mocap = Bvh(f.read())
        frame_rate = 1.0 / mocap.frame_time
    print('frame_rate: ', frame_rate)
    save_npz(output_file, smplx_trans, smplx_poses, gender='neutral', model_type='smplx', frame_rate=30)