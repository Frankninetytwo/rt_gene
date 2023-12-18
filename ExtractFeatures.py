import os


if __name__ == '__main__':

    os.system(
        'python3 rt_gene_standalone/estimate_gaze_standalone.py '
        ' InputImages '
        '--no-vis-headpose '
        '--no-vis-gaze '
        '--save-gaze '
        '--save-estimate '
        '--gaze_backend pytorch '
        '--output_path Output '
        '--models rt_gene/model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model '
        '--device-id-facedetection cpu:0'
        )