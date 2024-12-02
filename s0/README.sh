# Step1: near-field audio-only
bash run_asr_near.sh --stage 1 --stop_stage 6
# Step2: far-field audio-only
bash run_asr_far.sh --stage 1 --stop_stage 7
# Step3: video-only
bash run_vsr.sh --stage 1 --stop_stage 8
# Step3: audio+video
bash run_avsr.sh --stage 1 --stop_stage 3