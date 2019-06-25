import tensorflow as tf
import numpy as np
from utils import *
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple


BA_parameters = namedtuple('parameters',
                            'height,'
                            'width, '
                            'num_iters, '
                            'batch_size')

class BA_tf(object):

    def __init__(self, params):

        input_height = params.height
        input_width  = params.width
        batch_size = params.batch_size
        num_iters = params.num_iters

        self.intrinsics = tf.placeholder( tf.float32, [batch_size, 3, 3]  )

        self.T_ts = tf.placeholder( tf.float32, [batch_size, 6]  )
        self.Twc = pose_vec2mat(self.T_ts)  #

        self.src_rgb = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 1])
        self.tgt_rgb = tf.placeholder(tf.float32, [batch_size, input_height, input_width, 1])

        self.tgt_d = tf.placeholder(tf.float32, [batch_size, input_height, input_width , 1])


        # depth value validity check
        condition =  tf.greater_equal( self.tgt_d, 1)
        self.valid_depth_mask = tf.where ( condition , tf.ones_like(self.tgt_d) , tf.zeros_like(self.tgt_d)   )

        # image gradient mask, filter out low gradient pixels.
        self.tgt_gradient = img_gradient(self.tgt_rgb)
        condition =  tf.greater_equal( self.tgt_gradient, 40)
        self.valid_pixel_mask = tf.where ( condition, tf.ones_like (self.tgt_d) , tf.zeros_like(self.tgt_d)  )

        self.pixel_coords = meshgrid(batch_size, input_height, input_width)
        self.tgt_cam_coords = pixel_to_cam(self.tgt_d, self.pixel_coords, self.intrinsics)




    def pose_registration(self, num_iters=10 ):

        Twc = self.Twc

        for i in range(num_iters) :
            Twc = self.dense_BA_layer(Twc, self.src_rgb, self.tgt_rgb, self.tgt_cam_coords,
                                                   self.intrinsics)

        self.pose_update = Twc



    def gauss_newton_step(self,  J , residue , pose):
        ''' delta_xi = - (JTJ)^-1 * JTR
        :param J: [batch, width*height, 6 ]
        :param residue: [batch, width*height, 6 ]
        :param pose: [batch, 4, 4 ]
        :return: updated pose: [batch, 4, 4 ]
        '''
        batch_size, _, _ = J.get_shape().as_list()
        JTJ = tf.matmul ( tf.transpose( J ,perm=[0, 2, 1]) ,J )
        JTJ_inv = tf.matrix_inverse( JTJ )
        JTR = tf.matmul (  tf.transpose( J ,perm=[0, 2, 1]) , residue)
        delta_Xi = -1 * tf.matmul( JTJ_inv , JTR )
        delta_Xi =  tf.reshape ( delta_Xi , [ batch_size , -1])
        delta_T = expMap(delta_Xi)
        pose_update = tf.matmul (  delta_T ,pose )
        return pose_update


    def dense_BA_layer(self,current_pose, src_feature, tgt_feature , tgt_cam_coords, intrinsics):
        '''
        :param current_pose: [batch, 4, 4]
        :param src_feature: [batch, height, width, 1]
        :param tgt_feature: [batch, height, width, 1]
        :param tgt_cam_coords: [batch, 4, height, width]
        :param intrinsics: [batch, 3, 3]
        :param batch_size:
        :return:
        '''

        batch, height, width, _ = src_feature.get_shape().as_list()


        src_cam_coords = cam_to_cam( tgt_cam_coords, tf.matrix_inverse (current_pose))
        # src_cam_coords = cam_to_cam( tgt_cam_coords, current_pose)
        src_pixel_coords = cam_to_pixel( src_cam_coords, intrinsics)

        proj_feature  = bilinear_sampler(src_feature, src_pixel_coords)

        # check if the projection of a pixel is inside the src frame
        src_x = src_pixel_coords [:,:,:,0:1]
        src_y = src_pixel_coords [:,:,:,1: ]
        condition =tf.logical_and(
            tf.logical_and( tf.greater_equal(src_x, 4) ,  tf.less_equal ( src_x, width - 4 ) ),
            tf.logical_and( tf.greater_equal(src_y, 4) ,  tf.less_equal ( src_y, height - 4 ) )
        )
        in_frame_mask = tf.where(condition, tf.ones_like(self.valid_depth_mask), tf.zeros_like(self.valid_depth_mask))

        filter = self.valid_depth_mask * in_frame_mask  #* self.valid_pixel_mask

        residue =  (tgt_feature - proj_feature) * filter

        self.residue = residue

        residue = tf.reshape ( residue ,  [ batch  , -1 , 1 ] )
        J = compute_jacobian_residue_ksai( src_feature, src_pixel_coords, src_cam_coords, self.intrinsics,  filter)

        pose_update = self.gauss_newton_step ( J, residue, current_pose )

        self.curr_proj_image = proj_feature

        return pose_update



    def run(self, sess,  src_c, tgt_c, TD, pose,  cam):

        return sess.run([self.pose_update],
                        feed_dict={
                            self.src_rgb: src_c,
                            self.tgt_rgb: tgt_c,
                            self.tgt_d: TD ,
                            self.T_ts: pose,
                            self.intrinsics: cam
                        })



    def show_projection(self, sess,  src_c, tgt_c, TD, pose, cam):

        residue, curr_proj_image, tgt_rgb , src_rgb= sess.run([  self.residue, self.curr_proj_image, self.tgt_rgb, self.src_rgb],
                        feed_dict={
                            self.src_rgb: src_c,
                            self.tgt_rgb: tgt_c,
                            self.tgt_d: TD,
                            self.T_ts: pose,
                            self.intrinsics: cam
                        })

        combine = np.concatenate ( [curr_proj_image.squeeze(),tgt_rgb.squeeze()] ,axis=1)
        combine2 = np.concatenate ( [src_rgb.squeeze(), np.abs(residue.squeeze())] ,axis=1)
        combine = np.concatenate ( [ combine, combine2 ] ,  axis=0 ) / 255
        cv2.imshow("combine" , combine)
        cv2.waitKey( 20)




