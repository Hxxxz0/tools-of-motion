import sys
import numpy as np
from bvh import Bvh
from scipy.spatial.transform import Rotation as R
import math

JOINT_MAP = {
    # 'BVH joint name': 'SMPL joint index'
    'Hips': 0,
    'LeftUpLeg': 1,
    'RightUpLeg': 2,
    'Spine1': 3,
    'LeftLeg': 4,
    'RightLeg': 5,
    'Spine2': 6,
    'LeftFoot': 7,
    'RightFoot': 8,
    'Neck': 9,
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
    'FZLeftIndex1': 22,
    'FZRightIndex1': 23,
}

def bvh_to_smpl(bvh_file, n_frames=None):
    with open(bvh_file, 'r') as f:
        mocap = Bvh(f.read())

    if n_frames is None:
        num_frames = len(mocap.frames)
    else:
        num_frames = min(n_frames, len(mocap.frames))

    num_frames_downsampled = math.ceil(num_frames)

    smpl_poses = np.zeros((num_frames_downsampled, 72))
    smpl_trans = np.zeros((num_frames_downsampled, 3))
    
    bvh_joint_names = set(mocap.get_joints_names())
    # print(bvh_joint_names)

    rotation_correction = R.from_euler('XYZ', [0, 0, 0], degrees=True)

    for i in range(0, num_frames, 1):
        print('Processing frame {}/{}'.format(i, num_frames), end='\r')
        for joint_name, joint_index in JOINT_MAP.items():
            print(joint_name, joint_index)

            rotation = R.from_euler('ZXY', mocap.frame_joint_channels(i, joint_name, ['Zrotation', 'Xrotation', 'Yrotation']), degrees=True)

            if joint_name == 'Hips':
                # rotation = rotation * rotation_correction
                rotation = rotation_correction * rotation

            smpl_poses[i, 3 * joint_index:3 * (joint_index + 1)] = rotation.as_rotvec()

            if joint_name == 'Hips':
                x, y, z = mocap.frame_joint_channels(i, joint_name, ['Xposition', 'Yposition', 'Zposition'])
                smpl_trans[i] = np.array([x, -z, y])

    # mirror
    # smpl_trans[:, 1] *= -1

    scale_factor = 0.009
    smpl_trans *= scale_factor

    return smpl_trans, smpl_poses

import pickle

def save_pkl(output_file, smpl_trans, smpl_poses, gender='female', model_type='smpl', frame_rate=30, smpl_scaling=1.0):
    data = {
        'smpl_trans': smpl_trans.astype(np.float32),
        'smpl_poses': smpl_poses.astype(np.float32),
        'smpl_scaling' : np.array([smpl_scaling], dtype=np.float32), 
    }
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)



if __name__ == '__main__':
    # python bvh2smpl.py data/input/1.bvh data/test/1.pkl
    bvh_file = "1.bvh"
    output_file = "1.pkl"

    smpl_trans, smpl_poses = bvh_to_smpl(bvh_file, n_frames=2000)

    with open(bvh_file, 'r') as f:
        mocap = Bvh(f.read())
        frame_rate = 1.0 / mocap.frame_time
    print('frame_rate: ', frame_rate)
    save_pkl(output_file,smpl_trans,smpl_poses,frame_rate)
    
    