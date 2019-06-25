import tensorflow as tf
import numpy as np
from utils import *
import cv2
import matplotlib.pyplot as plt
from BA_tf import *


def load_freiburg_pair (   src_id, tgt_id, scale=1. ):

    cx = 325.5 /scale
    cy = 253.5 /scale
    fx = 518.0 /scale
    fy = 519.0 /scale
    depth_scale = 5000.0

    input_height =  int (480/scale)
    input_width = int (640/scale)


    intrin = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

    path_to_dataset='./test_data/rgbd_dataset_freiburg2_desk/' # MacOS
    associate_file = path_to_dataset + "associate.txt"

    def freiburgtxt2mat(lst):
        pose = [float(a.strip()) for a in lst]  # tx ty tz qx qy qz qw
        rot = quat2mat((pose[-1], pose[-4], pose[-3], pose[-2]))
        trn = np.asarray(pose[:3], dtype=np.float32).reshape([3, 1])
        T = np.concatenate([rot, trn], axis=-1)
        tile = np.asarray([[0, 0, 0, 1]], dtype=np.float32)
        T = np.concatenate([T, tile], axis=0)
        return T

    with open(associate_file , 'rb') as f :
        lines = f.readlines()

    src = lines[src_id].decode().split(' ')
    src_c = cv2.imread(  path_to_dataset + src[1] )
    src_d = cv2.imread( (path_to_dataset + src[3]).strip() , -1)/depth_scale
    src_d = cv2.resize(src_d, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
    src_c = cv2.cvtColor(src_c, cv2.COLOR_BGR2GRAY)
    src_c = cv2.resize(src_c, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
    src_pose = freiburgtxt2mat( src[-7:])
    # print  src_pose



    tgt = lines[tgt_id].decode().split(' ')
    tgt_c = cv2.imread(  path_to_dataset + tgt[1] )
    tgt_d = cv2.imread( (path_to_dataset + tgt[3]).strip() , -1)/depth_scale
    tgt_d = cv2.resize(tgt_d, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
    tgt_c = cv2.cvtColor(tgt_c, cv2.COLOR_BGR2GRAY)
    tgt_c = cv2.resize(tgt_c, (input_width, input_height), interpolation=cv2.INTER_CUBIC)
    tgt_pose = freiburgtxt2mat( tgt[-7:])
    # print tgt_pose


    T_ts = np.matmul( np.linalg.inv ( tgt_pose) , src_pose )
    ang_gt = mat2euler_np( T_ts [:3, :3])
    trn_gt = T_ts[:3, 3:].squeeze()
    gt_pose = np.concatenate ( [trn_gt, ang_gt] , axis=0 )   # tx, ty, tz, rx, ry, rz
    gt_pose = np.expand_dims( gt_pose, axis=0)
    # print gt_pose

    src_c = np.expand_dims(src_c , axis=0)
    src_c = np.expand_dims(src_c , axis=-1)
    tgt_c = np.expand_dims(tgt_c , axis=0)
    tgt_c = np.expand_dims(tgt_c , axis=-1)
    src_d = np.expand_dims( src_d, axis=0)
    src_d = np.expand_dims( src_d, axis=-1)
    tgt_d = np.expand_dims( tgt_d, axis=0)
    tgt_d = np.expand_dims( tgt_d, axis=-1)
    intrin = np.expand_dims(intrin , axis=0)
    return src_c, src_d, tgt_c, tgt_d, gt_pose, T_ts, intrin


if __name__ == '__main__':

    scale =1.
    input_height =  int (480/scale)
    input_width = int (640/scale)

    srd_id = 0
    tgt_id = 10


    src_c, src_d, tgt_c, tgt_d, gt_pose,T_ts, intrin = load_freiburg_pair( srd_id, tgt_id , scale=scale )


    params = BA_parameters(
        height=input_height,
        width=input_width,
        num_iters=0,
        batch_size=1
    )

    BA = BA_tf(params)
    BA.pose_registration( num_iters= 1)

    est_pose = np.asarray ( [[0,0,0,0,0,0]] , dtype=np.float32 )  # tx, ty, tz, rx, ry, rz

    with tf.Session() as sess:


        for iter in range ( 50 ):
            # print ("pose in", est_pose)

            BA.show_projection(sess, src_c, tgt_c, tgt_d, est_pose, intrin)

            pose_update = BA.run( sess, src_c , tgt_c, tgt_d, est_pose, intrin )[0]
            pose_update = pose_update.squeeze()
            ang_est = mat2euler_np(pose_update[:3, :3])
            trn_est = pose_update[:3, 3:].squeeze()
            est_pose= np.concatenate ( [trn_est, ang_est] , axis=0)
            est_pose= np.expand_dims ( est_pose, axis=0 )

        cv2.waitKey()
