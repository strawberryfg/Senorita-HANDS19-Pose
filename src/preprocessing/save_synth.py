# author: Anil Armagan
# contact: a.armagan@imperial.ac.uk
# date: 20/07/2019
# description: this file is provided for HANDS19 Challenge to render a synthetic image using the MANO model and the provided prarameters.
# usage: python3 render_mano.py --task-id=1 --frame-id=1 --mano-model-path=./MANO_RIGHT.pkl

import numpy as np
import cv2
from os.path import join, exists
import argparse

from utils.MANO_SMPL import MANO_SMPL
from utils.renderer import HandRenderer
from utils.crop import crop, pixel2world, world2pixel
from utils.reader import read_image, read_anno
from utils.vis import blend_frames, draw_joints, draw_bbox2d, copypaste_crop2org

# BigHand camera parameters
u0 = 315.944855
v0 = 245.287079
fx = 475.065948
fy = 475.065857
bbsize = 300.0  # mm
center_joint_id = 3  # Middle finger MCP joint in BigHand indexing


# make a few checks for the arguments and paths
def check_args(args):
    if(args.task_id == 1):
        print(args.frame_id)
        assert (args.frame_id < 175951) # 175951 images in task1
    elif (args.task_id == 2):
        assert (args.frame_id < 45212) # 45212 images in task2
    elif (args.task_id == 3):
        assert (args.frame_id < 1798) # highest frame-id in a sequence of task3 is 1798,
    else:
        assert (args.task_id not in [1, 2, 3]), 'Task-id should be one of the three tasks, [1,2,3]!'

    assert(exists(args.mano_model_path)), "MANO model file does not exist at %s. Please download the model from official MANO page." % args.mano_model_path
    assert(exists(args.joint_anno_path)), "3D joint ground truth annotations doesn't exist at %s." % args.joint_anno_path
    assert(exists(args.mano_anno_path)), "Fitted MANO model annotations doesn't exist at %s." % args.mano_anno_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HANDS19 MANO Renderer')
    parser.add_argument('--task-id', type=int, default=1, required=True, help='Task id of the challenge you want to see the rendering.  1, 2 or 3 for HANDS19')
    parser.add_argument('--frame-id', type=int, default=0, required=True, help='Frame id of the corresponding image, eg. frame_id=0 -> IMG_D00000000.png')
    parser.add_argument('--sequence-id', type=str, default='MC1', help='Sequence id for Task3') # invalid for now. Task3 processing will be added later.
    parser.add_argument('--mano-model-path', default='./MANO_RIGHT.pkl', help='Path to mano model file.')
    args = parser.parse_args()

    args.joint_anno_path = '../Task %d/training_joint_annotation.txt' % args.task_id
    args.mano_anno_path = '../Task %d/training_mano_annotation.txt' % args.task_id
    args.bbs_anno_path = '../Task %d/training_bbs_deneme.txt' % args.task_id
    args.frame_root_path = '../Task %d/training_images' % args.task_id

    check_args(args)

    # create MANO object, read model and initialize
    mano = MANO_SMPL(args.mano_model_path).cuda()
    renderer = HandRenderer(faces=mano.faces)

    mano_gt_path = '../Task %d/training_joint_annotation_synth.txt' % args.task_id
    f_mano_gt_path = open(mano_gt_path, 'w')

    for frame_id_ in range(121763, 175951):

        # read 3D joint annotations for the input frame
        #mano = MANO_SMPL(args.mano_model_path).cuda()
        frame_name, joints3d_anno = read_anno(frame_id=frame_id_, path=args.joint_anno_path)
        #print(frame_name)

        joints3d_anno = joints3d_anno.reshape(21,3)
        joints2d_anno = world2pixel(joints3d_anno[:,0], joints3d_anno[:,1], joints3d_anno[:,2], u0, v0, fx, fy)

        # read mano parameters for the input frame
        frame_name_chk, mano_anno = read_anno(frame_id=frame_id_, path=args.mano_anno_path)

        mano_cam = mano_anno[:4][np.newaxis]
        mano_quat = mano_anno[4 : 4+4][np.newaxis]
        mano_art = mano_anno[4+4 : 4+4+45][np.newaxis]
        mano_shape = mano_anno[4+4+45:][np.newaxis]

        assert (frame_name == frame_name_chk) # simple check to make sure mano params and joint annotations are loaded for the same frame name

        frame_path = join(args.frame_root_path, frame_name)

        assert(exists(frame_path)), "Frame doesn't exist at %s" % frame_path

        img = read_image(frame_path) # read depth image


        # crop input image with gt 3d annotations.
        # cropping is done by fitting a 3d bounding box of size bbsize around the MCP joint of the middle finger.
        cropped_img, joints2d_anno_crop, center2d_org, center2d_crop, bb2d, success_crop = crop(img, joints3d_anno, u0=u0, v0=v0, fx=fx, fy=fy, bbsize=bbsize, center_joint=center_joint_id, offset=30)
        crop_img_w = cropped_img.shape[1]
        crop_img_h = cropped_img.shape[0]

        # prepare renderer, rendered image is same size as the cropped image
        #renderer = HandRenderer(faces=mano.faces)

        renderer.init_buffers(image_w=crop_img_w, image_h=crop_img_h)

        # get mano vertices and joints with the current parameters
        vertices, joints_normed_ren = mano.get_mano_vertices(quat=mano_quat, pose=mano_art, shape=mano_shape, cam=mano_cam)


        joints2d_ren_crop = joints_normed_ren.copy()
        joints2d_ren_crop[:, 0] *= crop_img_w
        joints2d_ren_crop[:, 1] *= crop_img_h
        joints2d_ren_crop[:, 2] *= bbsize
        joints2d_ren_crop[:, 2] -= joints2d_ren_crop[center_joint_id, 2] # recenter around center joint's depth

        joints2d_ren_org = joints2d_ren_crop.copy()
        joints2d_ren_org[:,0] += center2d_org[0] - crop_img_h//2
        joints2d_ren_org[:,1] += center2d_org[1] - crop_img_w//2
        joints2d_ren_org[:,2] += center2d_org[2]

        

        # get xyz
        joints3d_ren_org = pixel2world(joints2d_ren_org[:, 0], joints2d_ren_org[:, 1], joints2d_ren_org[:, 2], u0, v0, fx, fy)

        # render normalized img
        
        rendered_img_crop = renderer.render_mano(vertices).copy()
        print("%4d haha" % frame_id_)

        # background set to 0 and unnormalize depth
        mask = rendered_img_crop == 1

        cropped_img = cropped_img * bbsize + joints2d_ren_crop[center_joint_id, 2]

        rendered_img_crop = rendered_img_crop * bbsize + joints2d_ren_crop[center_joint_id, 2]
        rendered_img_crop[mask] = 0

        # paste the rendered image back into original image resolution.
        # and rearrange the depth values wrt center joint's depth
        # cropped image could have been obtained with padding, get the valid area
        
        rendered_img_org = np.zeros(img.shape, img.dtype)
        rendered_img_org, success_paste = copypaste_crop2org(org_img=rendered_img_org, crop_img=rendered_img_crop, center2d_org=center2d_org)
        rendered_img_org[rendered_img_org>0] += center2d_org[2] - joints2d_ren_crop[center_joint_id, 2] # change depth

        print('Joints 2D GT', joints2d_anno)
        print('Joints 2D MANO', joints2d_ren_org)
        print('Joints 3D GT', joints3d_anno)
        print('Joints 3D MANO', joints3d_ren_org)
        f_mano_gt_path.write('%s ' % frame_name)
        for j in range(0, 21):
        	for k in range(0, 3):
        		f_mano_gt_path.write('%12.6f ' % joints3d_ren_org[j][k])
        f_mano_gt_path.write('\n')

        # visualize
        # blend real and synthetic image
        canvas = blend_frames(img, rendered_img_org, joints2d_crop=joints2d_anno, joints2d_ren=joints2d_ren_org)

        #Save synth
        
        img_synth=np.uint16(rendered_img_org)
        save_path = 'F:/hands19/Task 1/training_images_synth/image_D%08d.png' % frame_id_
        cv2.imshow("rgb_img_org", img_synth)
        cv2.waitKey(0)
        cv2.imwrite(save_path, img_synth)
        



        vis_rendered_org = draw_joints(rendered_img_org, joints2d=joints2d_ren_org, color=(0,0,255))
        vis_rendered_crop = draw_joints(rendered_img_crop, joints2d=joints2d_ren_crop, color=(0,0,255))
        vis_cropped = draw_joints(cropped_img, joints2d=joints2d_anno_crop, color=(255,0,0))
        canvas = draw_bbox2d(canvas, bb2d, color=(0,0,255))

        cv2.imshow('Cropped Hand Image', vis_cropped)
        cv2.imshow('Rendered MANO',vis_rendered_org)
        cv2.imshow('Rendered MANO Crop', vis_rendered_crop)
        cv2.imshow('Blended Image Real+Synth', canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("Rendered Mano model for Task#%d - %s" % (args.task_id, frame_name))
    f_mano_gt_path.close()
