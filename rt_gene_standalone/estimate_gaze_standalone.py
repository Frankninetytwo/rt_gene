#!/usr/bin/env python

# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from __future__ import print_function, division, absolute_import

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
from pathlib import Path

sys.path.append('rt_gene/src')
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix

script_path = os.path.dirname(os.path.realpath(__file__))


def load_camera_calibration(calibration_file):
    import yaml
    with open(calibration_file, 'r') as f:
        cal = yaml.safe_load(f)

    dist_coefficients = np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1, 5)
    camera_matrix = np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3, 3)

    return dist_coefficients, camera_matrix


def extract_eye_image_patches(subjects):
    for subject in subjects:
        le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c


def estimate_gaze(base_name, color_img, dist_coefficients, camera_matrix, args, output_images_path):
    
    # If there no person or more than 1 person found in the image I don't know whose gaze to use,
    # hence return nan in these cases.
    yaw_hat = math.nan
    pitch_hat = math.nan

    faceboxes = landmark_estimator.get_face_bb(color_img)
    if len(faceboxes) == 0:
        tqdm.write('Could not find faces in the image')
        return yaw_hat, pitch_hat

    subjects = landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)
    extract_eye_image_patches(subjects)

    input_r_list = []
    input_l_list = []
    input_head_list = []
    valid_subject_list = []
    
    for idx, subject in enumerate(subjects):
        if subject.left_eye_color is None or subject.right_eye_color is None:
            tqdm.write('Failed to extract eye image patches')
            continue

        success, rotation_vector, _ = cv2.solvePnP(landmark_estimator.model_points,
                                                   subject.landmarks.reshape(len(subject.landmarks), 1, 2),
                                                   cameraMatrix=camera_matrix,
                                                   distCoeffs=dist_coefficients, flags=cv2.SOLVEPNP_DLS)

        if not success:
            tqdm.write('Not able to extract head pose for subject {}'.format(idx))
            continue

        _rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
        _m = np.zeros((4, 4))
        _m[:3, :3] = _rotation_matrix
        _m[3, 3] = 1
        # Go from camera space to ROS space
        _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]
        roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
        roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

        phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)

        face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        head_pose_image = landmark_estimator.visualize_headpose_result(face_image_resized, (phi_head, theta_head))

        if args.vis_headpose:
            plt.axis("off")
            plt.imshow(cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB))
            plt.show()

        if args.save_headpose:
            # add idx to cope with multiple persons in one image
            cv2.imwrite(os.path.join(output_images_path, os.path.splitext(base_name)[0] + '_headpose_%s.jpg'%(idx)), head_pose_image)

        input_r_list.append(gaze_estimator.input_from_image(subject.right_eye_color))
        input_l_list.append(gaze_estimator.input_from_image(subject.left_eye_color))
        input_head_list.append([theta_head, phi_head])
        valid_subject_list.append(idx)


    if len(valid_subject_list) != 1:
        return yaw_hat, pitch_hat

    gaze_est = gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
                                                    inference_input_right_list=input_r_list,
                                                    inference_headpose_list=input_head_list)

    for subject_id, gaze, headpose in zip(valid_subject_list, gaze_est.tolist(), input_head_list):
        subject = subjects[subject_id]
        # Build visualizations
        r_gaze_img = gaze_estimator.visualize_eye_result(subject.right_eye_color, gaze)
        l_gaze_img = gaze_estimator.visualize_eye_result(subject.left_eye_color, gaze)
        s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)

        if args.vis_gaze:
            plt.axis("off")
            plt.imshow(cv2.cvtColor(s_gaze_img, cv2.COLOR_BGR2RGB))
            plt.show()

        if args.save_gaze:
            # add subject_id to cope with multiple persons in one image
            cv2.imwrite(os.path.join(output_images_path, os.path.splitext(base_name)[0] + '_gaze_%s.jpg'%(subject_id)), s_gaze_img)
            # cv2.imwrite(os.path.join(output_images_path, os.path.splitext(base_name)[0] + '_left.jpg'), subject.left_eye_color)
            # cv2.imwrite(os.path.join(output_images_path, os.path.splitext(base_name)[0] + '_right.jpg'), subject.right_eye_color)

        yaw_hat = gaze[1]
        pitch_hat = gaze[0]
    
    return yaw_hat, pitch_hat


# File will be written to
# CWD/Output/filename_of_video_without_file_extension.csv
# where filename_of_video_without_file_extension is a parameter of this function.
def write_estimated_gaze_to_file(filename_of_video_without_file_extension, timestamp_by_image_name, pitch_by_image_name, yaw_by_image_name):
    
    output_path = str(Path.cwd()) + '/Output/' + filename_of_video_without_file_extension + '.csv'
    
    with open(output_path, 'w') as f:
        
        f.write('frame,timestamp in s,success,yaw in radians,pitch in radians\n')

        # n-th frame of video is written to file with the name n.jpg. This name is used
        # as key to access the coressponding timestamp, yaw and pitch.
        for image_name in [str(i) for i in range(0, len(timestamp_by_image_name.keys()))]:
            f.write('{},{},{},{},{}\n'.format(
                int(image_name)+1,
                str(round(timestamp_by_image_name[image_name], 3)), # +/- 0.001 radians (less 0.1 degrees) can be rounded off (easier to compare output file to output from OpenFace)
                0 if math.isnan(yaw_by_image_name[image_name]) else 1, # pitch and yaw are set to nan whenever no person or more than 2 persons were detected in a video frame
                str(round(-yaw_by_image_name[image_name], 3)), # negate value to adjust to OpenFace output
                str(round(pitch_by_image_name[image_name], 3)))
                )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate gazes in a video using pretrained model')
    parser.add_argument('--video', type=str, dest='video_path', help='path of the video to proccess')
    parser.add_argument('--calib-file', type=str, dest='calib_file', default=None, help='Camera calibration file')
    parser.add_argument('--vis-headpose', dest='vis_headpose', action='store_true', help='Display the head pose images')
    parser.add_argument('--no-vis-headpose', dest='vis_headpose', action='store_false', help='Do not display the head pose images')
    parser.add_argument('--save-headpose', dest='save_headpose', action='store_true', help='Save the head pose images')
    parser.add_argument('--no-save-headpose', dest='save_headpose', action='store_false', help='Do not save the head pose images')
    parser.add_argument('--vis-gaze', dest='vis_gaze', action='store_true', help='Display the gaze images')
    parser.add_argument('--no-vis-gaze', dest='vis_gaze', action='store_false', help='Do not display the gaze images')
    parser.add_argument('--save-gaze', dest='save_gaze', action='store_true', help='Save the gaze images')
    parser.add_argument('--no-save-gaze', dest='save_gaze', action='store_false', help='Do not save the gaze images')
    parser.add_argument('--gaze_backend', choices=['tensorflow', 'pytorch'], default='tensorflow')
    parser.add_argument('--models', nargs='+', type=str, default=[os.path.abspath(os.path.join(script_path, '../rt_gene/model_nets/Model_allsubjects1.h5'))],
                        help='List of gaze estimators')
    parser.add_argument('--device-id-facedetection', dest="device_id_facedetection", type=str, default='cuda:0', help='Pytorch device id. Set to "cpu:0" to disable cuda')

    parser.set_defaults(vis_gaze=False)
    parser.set_defaults(save_gaze=False)
    parser.set_defaults(vis_headpose=False)
    parser.set_defaults(save_headpose=False)

    args = parser.parse_args()

    # !!!
    # !!!!
    # !!!!!
    # CAREFUL WHEN EDITING THIS PATH: Everything inside this folder gets deleted, so don't
    # assign a path to it that contains valuable data.
    image_folder = 'Frames'
    timestamp_by_image_name = dict()

    os.system('rm ' + image_folder + '/*')


    output_images_path = str(Path.cwd()) + '/OutputImages'

    if not os.path.isdir(output_images_path):
        os.makedirs(output_images_path)

    os.system('rm ' + output_images_path + '/*')


    video_capture = cv2.VideoCapture(args.video_path)
    frame_index = 0

    while True:

            success, frame = video_capture.read()

            if not success:
                # no further frames available
                break

            # cv2.CAP_PROP_POS_MSEC is sometimes very far off! For the final frame of an approx. 29.000 ms long video (.webm) it
            # returned me a timestamp of almost 35.000 ms!
            #timestamp_by_frame.append(video_capture.get(cv2.CAP_PROP_POS_MSEC))
            # ... So instead I'm going to estimate the timestamp like OpenFace does it
            # (see /OpenFace/lib/local/Utilities/src/SequenceCapture.cpp, line 457)
            current_timestamp = round(frame_index * (1.0 / video_capture.get(cv2.CAP_PROP_FPS)), 3)
            timestamp_by_image_name[str(frame_index)] = current_timestamp

            cv2.imwrite(image_folder + '/' + str(frame_index) + ".jpg", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1

    video_capture.release()


    image_path_list = []
    for image_file_name in sorted(os.listdir(image_folder)):
        if image_file_name.lower().endswith('.jpg') or image_file_name.lower().endswith('.png') or image_file_name.lower().endswith('.jpeg'):
            if '_gaze' not in image_file_name and '_headpose' not in image_file_name:
                image_path_list.append(image_file_name)

    tqdm.write('Loading networks')
    landmark_estimator = LandmarkMethodBase(device_id_facedetection=args.device_id_facedetection,
                                            checkpoint_path_face=os.path.abspath(os.path.join(script_path, "../rt_gene/model_nets/SFD/s3fd_facedetector.pth")),
                                            checkpoint_path_landmark=os.path.abspath(
                                                os.path.join(script_path, "../rt_gene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                            model_points_file=os.path.abspath(os.path.join(script_path, "../rt_gene/model_nets/face_model_68.txt")))

    if args.gaze_backend == "tensorflow":
        from rt_gene.estimate_gaze_tensorflow import GazeEstimator

        gaze_estimator = GazeEstimator("/gpu:0", args.models)
    elif args.gaze_backend == "pytorch":
        from rt_gene.estimate_gaze_pytorch import GazeEstimator

        gaze_estimator = GazeEstimator(args.device_id_facedetection, args.models)
    else:
        raise ValueError("Incorrect gaze_base backend, choices are: tensorflow or pytorch")
    

    
    # E.g. if gaze estimation is applied on the image named 23.jpg, then there is a key '23',
    # which points to the corresponding gaze estimation value (yaw resp. pitch). The
    # value is nan if the image contains 0 or more than one person. In that case I
    # don't know whose gaze to use.
    yaw_by_image_name = dict()
    pitch_by_image_name = dict()
    
    for image_file_name in tqdm(image_path_list):
        tqdm.write('Estimate gaze on ' + image_file_name)
        image = cv2.imread(os.path.join(image_folder, image_file_name))
        if image is None:
            tqdm.write('Could not load ' + image_file_name + ', skipping this image.')
            continue

        if args.calib_file is not None:
            _dist_coefficients, _camera_matrix = load_camera_calibration(args.calib_file)
        else:
            im_width, im_height = image.shape[1], image.shape[0]
            tqdm.write('WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
            _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array(
                [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])

        yaw_hat, pitch_hat = estimate_gaze(image_file_name, image, _dist_coefficients, _camera_matrix, args, output_images_path)

        yaw_by_image_name[os.path.splitext(image_file_name)[0]] = yaw_hat
        pitch_by_image_name[os.path.splitext(image_file_name)[0]] = pitch_hat
    
    #print('timestamp_by_image_name =', timestamp_by_image_name)
    #print('yaw_by_image_name =', yaw_by_image_name)
    #print('pitch_by_image_name =', pitch_by_image_name)
    
    write_estimated_gaze_to_file(Path(args.video_path).stem, timestamp_by_image_name, pitch_by_image_name, yaw_by_image_name)