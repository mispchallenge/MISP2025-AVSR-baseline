#!/bin/bash

# Copyright (c) 2023 USTC (Zhe Wang)
. ./path.sh || exit 1;
# . ./path_debug.sh || exit 1;

# Use this to control how many gpu you use, It's 1-gpu training if you specify
# just 1gpu, otherwise it's is multiple gpu training based on DDP in pytorch
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# The NCCL_SOCKET_IFNAME variable specifies which IP interface to use for nccl
# communication. More details can be found in
# https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# export NCCL_SOCKET_IFNAME=ens4f1
export NCCL_DEBUG=INFO
stage=1 # start from 0 if you need to start from data preparation
stop_stage=6

# The num of machines(nodes) for multi-machine training, 1 is for one machine.
# NFS is required if num_nodes > 1.
num_nodes=1

# The rank of each node or machine, which ranges from 0 to `num_nodes - 1`.
# You should set the node_rank=0 on the first machine, set the node_rank=1
# on the second machine, and so on.
node_rank=0

nj=16

# data_type can be `raw` or `shard`. Typically, raw is used for small dataset,
# `shard` is used for large dataset which is over 1k hours, and `shard` is
# faster on reading data and training.
data_type=raw
num_utts_per_shard=1000

train_set=train
train_config=pretrain/wenetspeech_u2pp_conformer_exp/train.yaml 
cmvn=true
dir=exp/asr_near_wenetspeech_v2/
tensor_dir=tensorboard/asr_near_wenetspeech/
enhancement_type=
checkpoint=pretrain/wenetspeech_u2pp_conformer_exp/final.pt
dict=pretrain/wenetspeech_u2pp_conformer_exp/units.txt

# use average_checkpoint will get better result
average_checkpoint=true
decode_checkpoint=$dir/final.pt
average_num=10  # Orignal:30
decode_modes="ctc_greedy_search ctc_prefix_beam_search attention attention_rescoring"

# MISP2025 Dataset Path
misp2025_corpus=/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt/data/MISP-Meeting/ # your data path
# Python Path
python_path=/home4/intern/minggao5/anaconda3/envs/zhewang18-wenet/bin/ # your python environment path
# Wenet_Path
wenet_misp2025_path=/train33/sppro/permanent/hangchen2/pandora/egs/misp2025 # your wenet path
wenet_misp2025_data_path=${wenet_misp2025_path}/s0/data_near_audio
# Decoding_Stage
max_epoch=60
decoding_chunk_size=
ctc_weight=0.5
reverse_weight=0.3

. tools/parse_options.sh || exit 1;

##########################################################################
# Step1: Generate 'wav.scp' and 'text_sentence' file
##########################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then   
  for x in training eval dev; do
    if [ -d ${wenet_misp2025_data_path}/${x}_near_audio ]; then
      rm -rf ${wenet_misp2025_data_path}/${x}_near_audio
      echo 'Exist old files are deleted.'
    fi

    for subdir in "${misp2025_corpus}"/${x}/*/
    do
      for folder in "$subdir"*/
      do
        if [[ $(basename "$folder") =~ F8N ]]; then
          ${python_path}python local/prepare_data.py -nj 1 $folder'*.wav' \
          $folder'*.TextGrid' ${wenet_misp2025_data_path}/${x}_near_audio|| exit 1;
        fi
      done
    done
  done
fi
##########################################################################
# Step2: Data segmentation
##########################################################################
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  for x in training dev eval; do
    data_dir=${wenet_misp2025_data_path}/${x}_near_audio
    segment_dir=${wenet_misp2025_data_path}/${x}_near_audio_segment/pt
    echo "============================================================"
    echo "segment $data_dir, store in $segment_dir"
    ${python_path}python tool_misp2022/segment_wav_to_pt.py -nj 10 $data_dir $segment_dir || exit 1
    cat $segment_dir/segment.log
  done
fi
##########################################################################
# Step3: Generate wenet preparation file
##########################################################################
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  for x in training dev eval; do
    segment_dir=${wenet_misp2025_data_path}/${x}_near_audio_segment
    cp ${wenet_misp2025_data_path}/${x}_near_audio/text_sentence ${segment_dir}/text
    ${python_path}python local/prepare_segment_data.py -nj 1 ${segment_dir}/pt/'*.pt' ${segment_dir} || exit 1;
    echo "Preparing data.list with wav.scp and text."
    ${python_path}python tools/make_raw_list_misp2022.py ${segment_dir}/wav.scp ${segment_dir}/text ${segment_dir}/data.list
  done
fi

##########################################################################
# Step4: Compute CMVN
##########################################################################
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  for x in training; do
    segment_dir=${wenet_misp2025_data_path}/${x}_near_audio_segment
    ${python_path}tools/compute_cmvn_stats_misp2022.py --num_workers 16 --train_config $train_config \
      --in_scp ${segment_dir}/wav.scp \
      --out_cmvn ${segment_dir}/global_cmvn
  done
fi

##########################################################################
# Step5: Training
##########################################################################
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  global_cmvn_path=${wenet_misp2025_data_path}/training_near_audio_segment/global_cmvn
  train_data_list=${wenet_misp2025_data_path}/training_near_audio_segment/data.list
  dev_data_list=${wenet_misp2025_data_path}/dev_near_audio_segment/data.list

  mkdir -p $dir
  # You have to rm `INIT_FILE` manually when you resume or restart a
  # multi-machine training.
  INIT_FILE=$dir/ddp_init

  init_method=file://$(readlink -f $INIT_FILE)
  echo "$0: init method is $init_method"
  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
  # Use "nccl" if it works, otherwise use "gloo"
  # dist_backend="gloo"
  dist_backend="nccl"
  world_size=`expr $num_gpus \* $num_nodes`
  echo "total gpus is: $world_size"
  cmvn_opts=
  $cmvn && cp ${global_cmvn_path} $dir
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  # train.py rewrite $train_config to $dir/train.yaml with model input
  # and output dimension, and $dir/train.yaml will be used for inference
  # and export.
  for ((i = 0; i < $num_gpus; ++i)); do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    # Rank of each gpu/process used for knowing whether it is
    # the master of a worker.
    rank=`expr $node_rank \* $num_gpus + $i`
    ${python_path}python wenet/bin/train_misp2022.py --gpu $gpu_id \
      --config $train_config \
      --data_type $data_type \
      --symbol_table $dict \
      --train_data ${train_data_list} \
      --cv_data ${dev_data_list} \
      ${checkpoint:+--checkpoint $checkpoint} \
      --tensorboard_dir $tensor_dir \
      --model_dir $dir \
      --ddp.init_method $init_method \
      --ddp.world_size $world_size \
      --ddp.rank $rank \
      --ddp.dist_backend $dist_backend \
      --num_workers 1 \
      $cmvn_opts \
      --pin_memory
  } &
  done
  wait
fi


##########################################################################
# Step6: Decoding
##########################################################################
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  eval_data_list=${wenet_misp2025_data_path}/eval_near_audio_segment/data.list
  eval_text=${wenet_misp2025_data_path}/eval_near_audio_segment/text
  $cmvn && cmvn_opts="--cmvn ${dir}/global_cmvn"

  # Test model, please specify the model you want to test by --checkpoint
  if [ ${average_checkpoint} == true ]; then
    decode_checkpoint=$dir/avg_${average_num}.pt
    echo "do model average and final checkpoint is $decode_checkpoint"
    ${python_path}python wenet/bin/average_model.py \
      --dst_model $decode_checkpoint \
      --src_path $dir  \
      --num ${average_num} \
      --max_epoch ${max_epoch} \
      --val_best 
  fi
  # Please specify decoding_chunk_size for unified streaming and
  # non-streaming model. The default value is -1, which is full chunk
  # for non-streaming inference.
  for mode in ${decode_modes}; do
  {
    gpu_id=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f$[$i+1])
    test_dir=$dir/eval_${mode}_${decoding_chunk_size}
    mkdir -p $test_dir
    ${python_path}python wenet/bin/recognize_misp2022.py --gpu $gpu_id \
      --mode $mode \
      --config $train_config \
      --data_type $data_type \
      --test_data ${eval_data_list} \
      --checkpoint $decode_checkpoint \
      --beam_size 10 \
      --batch_size 1 \
      --penalty 0.0 \
      --dict $dict \
      --ctc_weight $ctc_weight \
      --reverse_weight $reverse_weight \
      $cmvn_opts \
      --result_file $test_dir/text \
      ${decoding_chunk_size:+--decoding_chunk_size $decoding_chunk_size}
    ${python_path}python tools/compute-wer.py --char=1 --v=1 \
      ${eval_text} $test_dir/text > $test_dir/wer
  } &
  done
  wait
fi

