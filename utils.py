from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def skew(phi):
    '''
    :param phi: [batch, 3]
    :return: Phi : [batch, 3, 3 ]
    '''
    batch, _ = phi.get_shape().as_list()
    phi_0 = tf.slice(phi, [0, 0], [-1, 1])
    phi_1 = tf.slice(phi, [0, 1], [-1, 1])
    phi_2 = tf.slice(phi, [0, 2], [-1, 1])
    zero = tf.zeros_like(phi_0)
    Phi = tf.concat([zero, -phi_2, phi_1, phi_2, zero, -phi_0, -phi_1, phi_0, zero], axis=-1)
    Phi = tf.reshape(Phi, [batch, 3, 3])
    return Phi

def expMap( ksai):
    '''exponetial mapping.
    :param ksai: [Batch, 6]
    :return: SE3: [Batch, 4, 4]
    '''
    batch, _ = ksai.get_shape().as_list()
    omega = tf.slice(ksai, [0, 0], [-1, 3])  # rot
    upsilon = tf.slice(ksai, [0, 3], [-1, -1])  # trn
    theta = tf.norm(omega, 2, axis=-1, keep_dims=True)
    theta = tf.expand_dims(theta, axis=-1)
    theta = tf.tile(theta, multiples=[1, 3, 3])
    Omega = skew(omega)
    Omega2 = tf.matmul(Omega, Omega)
    identities = tf.tile([tf.eye(3)], multiples=[batch, 1, 1])
    R = identities + \
        tf.sin(theta) * Omega / theta + \
        (1 - tf.cos(theta)) * Omega2 / (theta * theta)
    V = identities + \
        (1 - tf.cos(theta)) * Omega / (theta * theta) + \
        (theta - tf.sin(theta)) * Omega2 / (theta * theta * theta)
    t = tf.matmul(V, tf.expand_dims(upsilon, -1))
    T34 = tf.concat([R, t], axis=-1)
    brow = tf.tile([[[0., 0., 0., 1.]]], multiples=[batch, 1, 1])
    T44 = tf.concat([T34, brow], axis=1)
    return T44

def logMap(SE3):
    '''logarithmic mapping
    :param SE3: [B, 4, 4]
    :return: ksai [B, 6], trn after rot
    '''
    def deltaR (R):
        '''
        :param R:  [B, 3, 3]
        :return:
        '''
        v_0 = R[:,2,1] - R[:,1,2]
        v_1 = R[:,0,2] - R[:,2,0]
        v_2 = R[:,1,0] - R[:,0,1]
        v = tf.stack( [ v_0, v_1, v_2 ],axis= -1)
        return v

    batch, _, _ = SE3.get_shape().as_list()
    _R = SE3[:, :3, :3 ]
    _t = SE3[:, :3, 3:]
    d = 0.5 * ( _R[:,0,0] + _R [:,1,1] + _R[:,2,2]  -1 )
    d = tf.expand_dims ( d , axis= -1 )
    dR = deltaR ( _R )
    theta = tf.acos( d )
    omega = theta * dR / ( 2* tf.sqrt (1-d*d) )
    Omega = skew( omega )
    identities = tf.tile([tf.eye(3)], multiples=[batch, 1, 1])
    V_inv = identities - 0.5 * Omega + \
            ( 1 - theta / ( 2 * tf.tan ( theta/2 ) ) ) * tf.matmul( Omega , Omega ) / (theta * theta)
    upsilon = tf.matmul ( V_inv , _t)
    upsilon = tf.reshape ( upsilon , [batch, -1 ])
    ksai = tf.concat ( [ omega , upsilon ] , axis= -1  )
    return ksai

def cam_to_pixel( cam_coords, intrinsics):
    '''Transforms coordinates in a camera frame to the pixel frame.
    :param cam_coords: [batch, 4, height, width]
    :param intrinsics: [batch, 3, 3]
    :return: Pixel coordinates projected from the camera frame [batch, height, width, 2]
    '''
    batch, _, height, width = cam_coords.get_shape().as_list()
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch, 1, 1])
    intrinsics = tf.concat([intrinsics, tf.zeros([batch, 3, 1])], axis=2)
    intrinsics = tf.concat([intrinsics, filler], axis=1)
    world_coords = tf.reshape(cam_coords, [batch, 4, -1])
    unnormalized_pixel_coords = tf.matmul(intrinsics, world_coords)
    x_u = tf.slice(unnormalized_pixel_coords, [0, 0, 0], [-1, 1, -1])
    y_u = tf.slice(unnormalized_pixel_coords, [0, 1, 0], [-1, 1, -1])
    z_u = tf.slice(unnormalized_pixel_coords, [0, 2, 0], [-1, 1, -1])
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
    pixel_coords = tf.concat([x_n, y_n], axis=1)
    pixel_coords = tf.reshape(pixel_coords, [batch, 2, height, width])
    return tf.transpose(pixel_coords, perm=[0, 2, 3, 1])

def cam_to_cam(cam_1_coords, pose):
    '''Transform 3D coordinates from cam 1 to cam 2
    :param cam_1_coords: [batch, 4, height, width]
    :param pose: [batch, 4, 4]
    :return: cam coordinates in cam 2 [batch, 4, height, width]
    '''
    batch, _, height, width = cam_1_coords.get_shape().as_list()
    cam_1_coords = tf.reshape(cam_1_coords, [batch, 4, -1])
    cam_2_coords = tf.matmul(pose, cam_1_coords)
    cam_2_coords = tf.reshape(cam_2_coords, [batch, 4, height, width])
    return cam_2_coords


def pixel_to_cam( depth, pixel_coords, intrinsics, is_homogeneous=True):
    '''Transforms coordinates in the pixel frame to the camera frame. Camera model: [X,Y,Z]^T = D * K^-1 * [u,v,1]^T
    :param depth: [batch, height, width, 1]
    :param pixel_coords: homogeneous [batch, 3, height, width]
    :param intrinsics: [batch, 3, 3]
    :param is_homogeneous: return in homogeneous coordinates
    :return: Cam Coords [batch, 3 (4 if homogeneous), height, width]
    '''
    batch, height, width, _ = depth.get_shape().as_list()
    depth = tf.reshape(depth, [batch, 1, -1])
    pixel_coords = tf.reshape(pixel_coords, [batch, 3, -1])
    cam_coords = tf.matmul(tf.matrix_inverse(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = tf.ones([batch, 1, height * width])
        cam_coords = tf.concat([cam_coords, ones], axis=1)
    cam_coords = tf.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords


def compute_jacobian_residue_ksai( src_feature, src_pixel_coords, src_cam_coords, intrinsics, filter):
    '''compute jacobian of xi w.r.t residue.
    :param src_feature: [batch, height, width, 1]
    :param src_pixel_coords: [batch, height, width, 2]
    :param src_cam_coords: [batch, 4,  height, width]
    :param intrinsics: [batch, 3, 3]
    :param filter: [batch, height, width, 1]
    :return: Jacobian:  [batch, height*width, 6]
    '''

    batch, _, height, width = src_cam_coords.get_shape().as_list()

    X = tf.slice(src_cam_coords, [0, 0, 0, 0], [-1, 1, -1, -1])  # B , 1 , H, W
    Y = tf.slice(src_cam_coords, [0, 1, 0, 0], [-1, 1, -1, -1])
    Z = tf.slice(src_cam_coords, [0, 2, 0, 0], [-1, 1, -1, -1])

    fx = tf.slice(intrinsics, [0, 0, 0], [-1, 1, 1])  # B, 1, 1
    fx = tf.tile(fx, multiples=[1, height, width])  # B H W
    fx = tf.reshape(fx, [batch, 1, height, width])  # B 1 H W

    fy = tf.slice(intrinsics, [0, 1, 1], [-1, 1, 1])
    fy = tf.tile(fy, multiples=[1, height, width])
    fy = tf.reshape(fy, [batch, 1, height, width])

    invZ = 1. / (Z + 1e-10)
    invZ_2 = invZ * invZ

    # compute partial from ksai to uv
    # 6 dof in the (rot, trn) order
    partial_u_ksai_0 = - fx * X * Y * invZ_2
    partial_u_ksai_1 = fx + fx * X * X * invZ_2
    partial_u_ksai_2 = - fx * Y * invZ
    partial_u_ksai_3 = fx * invZ
    partial_u_ksai_4 = tf.zeros_like(partial_u_ksai_0)
    partial_u_ksai_5 = - fx * X * invZ_2
    partial_u_ksai = tf.stack([partial_u_ksai_0,
                               partial_u_ksai_1,
                               partial_u_ksai_2,
                               partial_u_ksai_3,
                               partial_u_ksai_4,
                               partial_u_ksai_5], axis=2)
    partial_v_ksai_0 = - fy - fy * Y * Y * invZ_2
    partial_v_ksai_1 = fy * X * Y * invZ_2
    partial_v_ksai_2 = fy * X * invZ
    partial_v_ksai_3 = tf.zeros_like(partial_v_ksai_0)
    partial_v_ksai_4 = fy * invZ
    partial_v_ksai_5 = - fy * Y * invZ_2
    partial_v_ksai = tf.stack([partial_v_ksai_0,
                               partial_v_ksai_1,
                               partial_v_ksai_2,
                               partial_v_ksai_3,
                               partial_v_ksai_4,
                               partial_v_ksai_5], axis=2)
    partial_uv_ksai = tf.concat([partial_u_ksai, partial_v_ksai], axis=1)  # b 2 6 h w
    partial_uv_ksai = tf.transpose(partial_uv_ksai, perm=[0, 3, 4, 1, 2])

    # compute image gradient
    right_pixel_coords = src_pixel_coords + tf.tile([[[[1., 0.]]]], multiples=[batch, height, width, 1])
    left_pixel_coords = src_pixel_coords + tf.tile([[[[-1., 0.]]]], multiples=[batch, height, width, 1])
    bot_pixel_coords = src_pixel_coords + tf.tile([[[[0., 1.]]]], multiples=[batch, height, width, 1])
    top_pixel_coords = src_pixel_coords + tf.tile([[[[0., -1.]]]], multiples=[batch, height, width, 1])
    partial_I_u = (bilinear_sampler(src_feature, right_pixel_coords)
                   - bilinear_sampler(src_feature, left_pixel_coords)) / 2
    partial_I_v = (bilinear_sampler(src_feature, bot_pixel_coords)
                   - bilinear_sampler(src_feature, top_pixel_coords)) / 2
    partial_I_uv = tf.concat([partial_I_u, partial_I_v], axis=-1)  # b h w 2
    partial_I_uv = tf.expand_dims(partial_I_uv, axis=3)

    # combine the gradient chain
    jacobian_res_ksai = tf.matmul(partial_I_uv, partial_uv_ksai)

    # filter out invalid residue
    jacobian_res_ksai = tf.reshape(jacobian_res_ksai, [batch, height * width, 6])
    filter = tf.reshape(filter, [batch, height * width, 1])
    filter = tf.tile(filter, multiples=[1, 1, 6])
    jacobian_res_ksai = jacobian_res_ksai * filter

    return jacobian_res_ksai


def img_gradient(img):
    '''compute image gradient
    :param img: [B, height, width, 1]
    :return: gradient: [B, height, width, 1]
    '''
    B, H, W, C = img.get_shape().as_list()
    top_pixel = img[:, :-2, :, :]
    bot_pixel = img[:, 2: , :, :]
    left_pixel = img[:, :, :-2, :]
    rigt_pixel = img[:, :,  2:, :]

    dy = bot_pixel - top_pixel
    dx = rigt_pixel - left_pixel

    aRow = tf.ones( [B,1,W, C], dtype= tf.float32 )
    aCol = tf.ones( [B,H,1, C], dtype= tf.float32 )

    dy= tf.concat ( [ aRow, dy, aRow ] , axis=1 )
    dx= tf.concat ( [ aCol, dx, aCol ], axis=2)
    dxy = tf.concat ( [dy, dx] , axis= -1)
    dxy = tf.norm ( dxy , axis= -1 )
    return  tf.expand_dims ( dxy , axis= -1 )


def mat2euler_np(rot_M):
    '''Conver rotation matrix to euler angle X->Y->Z, numpy version
    :param rot_M:
    :return:
    '''
    r11 = rot_M[0][0]
    r12 = rot_M[0][1]
    r13 = rot_M[0][2]

    r21 = rot_M[1][0]
    r22 = rot_M[1][1]
    r23 = rot_M[1][2]

    r31 = rot_M[2][0]
    r32 = rot_M[2][1]
    r33 = rot_M[2][2]

    rx = np.arctan2(-r23, r33)
    cy = np.sqrt(r11 * r11 + r12 * r12)
    ry = np.arctan2(r13, cy)
    rz = np.arctan2(-r12, r11)

    # return  tf.stack( [ rx, ry, rz ] )
    return np.stack([rx, ry, rz])

def quat2mat(q):
    '''
    :param q:
    :return:
    '''

    w, x, y, z = q
    Nq = w * w + x * x + y * y + z * z
    if Nq < 1e-8:
        return np.eye(3)
    s = 2.0 / Nq
    X = x * s
    Y = y * s
    Z = z * s
    wX = w * X
    wY = w * Y
    wZ = w * Z
    xX = x * X
    xY = x * Y
    xZ = x * Z
    yY = y * Y
    yZ = y * Z
    zZ = z * Z
    return np.array(
    [[1.0 - (yY + zZ), xY - wZ, xZ + wY],
     [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
     [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])



def euler2mat(z, y, x):
    '''Converts euler angles to rotation matrix
    :param z: rotation angle along z axis (in radians) -- size = [B, N]
    :param y: rotation angle along y axis (in radians) -- size = [B, N]
    :param x: rotation angle along x axis (in radians) -- size = [B, N]
    :return: Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    '''

    B = tf.shape(z)[0]
    N = 1
    z = tf.clip_by_value(z, -np.pi, np.pi)
    y = tf.clip_by_value(y, -np.pi, np.pi)
    x = tf.clip_by_value(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = tf.expand_dims(tf.expand_dims(z, -1), -1)
    y = tf.expand_dims(tf.expand_dims(y, -1), -1)
    x = tf.expand_dims(tf.expand_dims(x, -1), -1)

    zeros = tf.zeros([B, N, 1, 1])
    ones  = tf.ones([B, N, 1, 1])

    cosz = tf.cos(z)
    sinz = tf.sin(z)
    rotz_1 = tf.concat([cosz, -sinz, zeros], axis=3)
    rotz_2 = tf.concat([sinz,  cosz, zeros], axis=3)
    rotz_3 = tf.concat([zeros, zeros, ones], axis=3)
    zmat = tf.concat([rotz_1, rotz_2, rotz_3], axis=2)

    cosy = tf.cos(y)
    siny = tf.sin(y)
    roty_1 = tf.concat([cosy, zeros, siny], axis=3)
    roty_2 = tf.concat([zeros, ones, zeros], axis=3)
    roty_3 = tf.concat([-siny,zeros, cosy], axis=3)
    ymat = tf.concat([roty_1, roty_2, roty_3], axis=2)

    cosx = tf.cos(x)
    sinx = tf.sin(x)
    rotx_1 = tf.concat([ones, zeros, zeros], axis=3)
    rotx_2 = tf.concat([zeros, cosx, -sinx], axis=3)
    rotx_3 = tf.concat([zeros, sinx, cosx], axis=3)
    xmat = tf.concat([rotx_1, rotx_2, rotx_3], axis=2)

    xy = tf.matmul( xmat, ymat)
    rotMat = tf.matmul(xy, zmat)
    return rotMat

def pose_vec2mat(vec):
    '''
    :param vec:
    :return:
    '''
    batch_size, _ = vec.get_shape().as_list()
    translation = tf.slice(vec, [0, 0], [-1, 3])
    translation = tf.expand_dims(translation, -1)
    rx = tf.slice(vec, [0, 3], [-1, 1])
    ry = tf.slice(vec, [0, 4], [-1, 1])
    rz = tf.slice(vec, [0, 5], [-1, 1])
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = tf.squeeze(rot_mat, axis=[1])
    filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
    filler = tf.tile(filler, [batch_size, 1, 1])
    transform_mat = tf.concat([rot_mat, translation], axis=2)
    transform_mat = tf.concat([transform_mat, filler], axis=1)
    return transform_mat


def meshgrid(batch, height, width, is_homogeneous=True):
    '''Construct a 2D meshgrid.
    :param batch: batch size
    :param height: height of the grid
    :param width: width of the grid
    :param is_homogeneous: whether to return in homogeneous coordinates
    :return: x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    '''
    x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                    tf.transpose(tf.expand_dims(
                        tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
    y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                    tf.ones(shape=tf.stack([1, width])))
    x_t = (x_t + 1.0) * 0.5 * tf.cast(width - 1, tf.float32)
    y_t = (y_t + 1.0) * 0.5 * tf.cast(height - 1, tf.float32)
    if is_homogeneous:
        ones = tf.ones_like(x_t)
        coords = tf.stack([x_t, y_t, ones], axis=0)
    else:
        coords = tf.stack([x_t, y_t], axis=0)
    coords = tf.tile(tf.expand_dims(coords, 0), [batch, 1, 1, 1])
    return coords


def bilinear_sampler(imgs, coords):
    ''' Code borrowed from SFMLeaner (Zhou+ CVPR2017)
    Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    :param imgs: [batch, height, width, channels]
    :param coords: [batch, height, width, 2]
    :return: sampled image [batch, height, width, channels]
    '''

    def _repeat(x, n_repeats):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.stack([
                n_repeats,
            ])), 1), [1, 0])
        rep = tf.cast(rep, 'float32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

    with tf.name_scope('image_sampling'):
        coords_x, coords_y = tf.split(coords, [1, 1], axis=3)
        inp_size = imgs.get_shape()
        coord_size = coords.get_shape()
        out_size = coords.get_shape().as_list()
        out_size[3] = imgs.get_shape().as_list()[3]

        coords_x = tf.cast(coords_x, 'float32')
        coords_y = tf.cast(coords_y, 'float32')

        x0 = tf.floor(coords_x)
        x1 = x0 + 1
        y0 = tf.floor(coords_y)
        y1 = y0 + 1

        y_max = tf.cast(tf.shape(imgs)[1] - 1, 'float32')
        x_max = tf.cast(tf.shape(imgs)[2] - 1, 'float32')
        zero = tf.zeros([1], dtype='float32')

        x0_safe = tf.clip_by_value(x0, zero, x_max)
        y0_safe = tf.clip_by_value(y0, zero, y_max)
        x1_safe = tf.clip_by_value(x1, zero, x_max)
        y1_safe = tf.clip_by_value(y1, zero, y_max)

        ## bilinear interp weights, with points outside the grid having weight 0
        # wt_x0 = (x1 - coords_x) * tf.cast(tf.equal(x0, x0_safe), 'float32')
        # wt_x1 = (coords_x - x0) * tf.cast(tf.equal(x1, x1_safe), 'float32')
        # wt_y0 = (y1 - coords_y) * tf.cast(tf.equal(y0, y0_safe), 'float32')
        # wt_y1 = (coords_y - y0) * tf.cast(tf.equal(y1, y1_safe), 'float32')

        wt_x0 = x1_safe - coords_x
        wt_x1 = coords_x - x0_safe
        wt_y0 = y1_safe - coords_y
        wt_y1 = coords_y - y0_safe

        ## indices in the flat image to sample from
        dim2 = tf.cast(inp_size[2], 'float32')
        dim1 = tf.cast(inp_size[2] * inp_size[1], 'float32')
        base = tf.reshape(
            _repeat(
                tf.cast(tf.range(coord_size[0]), 'float32') * dim1,
                coord_size[1] * coord_size[2]),
            [out_size[0], out_size[1], out_size[2], 1])

        base_y0 = base + y0_safe * dim2
        base_y1 = base + y1_safe * dim2
        idx00 = tf.reshape(x0_safe + base_y0, [-1])
        idx01 = x0_safe + base_y1
        idx10 = x1_safe + base_y0
        idx11 = x1_safe + base_y1

        ## sample from imgs
        imgs_flat = tf.reshape(imgs, tf.stack([-1, inp_size[3]]))
        imgs_flat = tf.cast(imgs_flat, 'float32')
        im00 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx00, 'int32')), out_size)
        im01 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx01, 'int32')), out_size)
        im10 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx10, 'int32')), out_size)
        im11 = tf.reshape(tf.gather(imgs_flat, tf.cast(idx11, 'int32')), out_size)

        w00 = wt_x0 * wt_y0
        w01 = wt_x0 * wt_y1
        w10 = wt_x1 * wt_y0
        w11 = wt_x1 * wt_y1

        output = tf.add_n([
            w00 * im00, w01 * im01,
            w10 * im10, w11 * im11
        ])
        return output