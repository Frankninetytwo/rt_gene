import os
import argparse


def parse_args():
    
    parser = argparse.ArgumentParser(description='Estimate gazes in a video using pretrained model')
    
    parser.add_argument(
        '--video',
        dest='video_path',
        help='path of the video to proccess',  
        type=str
        )
    
    parser.add_argument(
        '--timestamp-to-start-at',
        dest='timestamp_to_start_at',
        help='point in time of the video from which on gaze shall be estimated (in seconds)',  
        type=float,
        default=0.0
    )
    
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if not args.video_path:
        print('argument --video required')
        exit()
    
    if not os.path.isfile(args.video_path):
        print('--video argument invalid: \"' + args.video_path + '\" is not a file!')
        exit()

    os.system(
        'python3 rt_gene_standalone/estimate_gaze_standalone.py '
        '--video ' + args.video_path + ' '
        '--timestamp-to-start-at ' + str(args.timestamp_to_start_at) + ' '
        '--no-vis-headpose '
        '--no-vis-gaze '
        '--gaze_backend pytorch '
        '--models rt_gene/model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model '
        '--device-id-facedetection cpu:0'
        )