# Required files

## Pre-trained models

For wenetspeech conformer, see https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md

For lrw_resnet18_dctcn_video_boundary.pth, see: https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/tree/master

## Lip detection results

We provide shoulder and lip detection results in  <a href="https://pan.baidu.com/s/1Xtis2q6xkYrqZoudJ4DDTA?pwd=bhyd" title="video_shoulder_lip_detection">video_shoulder_lip_detection</a> (password: bhyd). Please note that the detection results we provide are based on video segments that have been aligned through timestamp cropping (not from the original video). Due to algorithmic detection errors, we do not guarantee the accuracy of the detection results.


# Pipeline

## ASR

We first use WenetSpeech as the pretrained model to train near-field audio **(run_near.sh)**, and then we use the trained model as a pretrained model to train on far-field audio **(run_far.sh)**. Note that we perform GSS processing on far-field audio before training.

## AVSR

We use lrw_resnet18_dctcn_video_boundary.pth as the pretrained model to train a single video model **(run_vsr.sh)**. The resulting model serves as the pretrained model for the AVSR video modality. At the same time, we use the previously trained near-field audio model as the pretrained model for the AVSR audio modality. This setup is used to train the audio-visual model **(run_avsr.sh)**.

