#!/usr/bin/env bash

gss_path=$1
cur_path=$(pwd)
cd ${gss_path}
. ./cmd.sh
. ./path.sh
set -eou pipefail
cmd=run.pl
set -e

. ./utils/parse_options.sh
start=1
stop=4
nj=1
dataroot=$2
Manifest_root=$3
dset=$4
wenet_misp2025_data_path=$5


if [ $start -le 1 ] && [ $stop -ge 1 ]; then 
    echo "Stage 1_1: create $dset mainfest"   
    Manifest_dir=$Manifest_root/${dset}_wave/far/wpe/gss_new
    EXP_DIR=$Manifest_dir
    [[ ! -e $Manifest_dir ]] && mkdir -p $Manifest_dir
    python ${gss_path}/prepare_misp.py  \
    --data_dir $dataroot/${dset}_far \
    --sampling_rate 16000 \
    --manifest_dir $Manifest_dir \
    --channels 8

    echo "Stage 1_2: generate cuts"
    lhotse cut simple  \
    --force-eager \
    -r $Manifest_dir/recordings.jsonl.gz \
    -s $Manifest_dir/supervisions.jsonl.gz \
    $Manifest_dir/cuts.jsonl.gz

    echo "Stage 1_3: trim cuts"
    lhotse cut trim-to-supervisions --discard-overlapping --keep-all-channels \
    $EXP_DIR/cuts.jsonl.gz $EXP_DIR/cuts_per_segment.jsonl.gz
    echo "stage 1_4: clean cuts"
    python ${gss_path}/prepare_misp.py --mode filter_cut --cutpath $EXP_DIR/cuts_per_segment.jsonl.gz    # 注意目录
    echo "Stage 1_5: Split segments into $nj parts"
    gss utils split $nj $EXP_DIR/cuts_per_segment.jsonl.gz $EXP_DIR/split$nj
fi

  
if [ $start -le 2 ] && [ $stop -ge 2 ]; then
    cu_index=0
    echo "Stage 2: Runing GPU GSS"
    Manifest_dir=$Manifest_root/${dset}_wave/far/wpe/gss_new
    EXP_DIR=$Manifest_dir

    CUDA_VISIBLE_DEVICES=3 $cmd JOB=1 $EXP_DIR/log/enhance.JOB.log \
    gss enhance cuts \
        $EXP_DIR/cuts.jsonl.gz $EXP_DIR/split$nj/cuts_per_segment.JOB.jsonl.gz \
        $EXP_DIR/enhanced \
        --min-segment-length 0.1 \
        --max-segment-length 20.0 \
        --context-duration 15.0 \
        --use-garbage-class \
        --bss-iterations 20 \
        --max-batch-duration 10.0 \
        --num-buckets 3 \
        --num-workers 6 \
        --duration-tolerance 3.0 \
        --max-batch-cuts 1 \
        --force-overwrite \
        --enhanced-manifest $EXP_DIR/split$nj/cuts_enhanced.JOB.jsonl.gz
fi

if [ $start -le 3 ] && [ $stop -ge 3 ]; then
    echo "Stage 3: flac->wav"
    source_dir=${Manifest_root}/${dset}_wave/far/wpe/gss_new/enhanced
    target_dir=${wenet_misp2025_data_path}/${dset}_far_audio_segment/enhanced
    [[ ! -e $target_dir ]] && mkdir -p $target_dir
    python ${cur_path}/local/flac2wav.py ${source_dir} ${target_dir}
fi

if [ $start -le 4 ] && [ $stop -ge 4 ]; then
    echo "$dset: cuts.js.gz to dumpdir"
    Manifest_dir=${wenet_misp2025_data_path}/${dset}_far_audio_segment
    EXP_DIR=$Manifest_dir
    manifest_file=gssgpu_${dset}_far_wave

    for ((i=1; i<=nj; i++))
    do
      cp -R ${Manifest_root}/${dset}_wave/far/wpe/gss_new/split$i ${wenet_misp2025_data_path}/${dset}_far_audio_segment/split$i
    done
    cp ${Manifest_root}/${dset}_wave/far/wpe/gss_new/cuts.jsonl.gz ${wenet_misp2025_data_path}/${dset}_far_audio_segment/cuts.jsonl.gz
    cp ${Manifest_root}/${dset}_wave/far/wpe/gss_new/cuts_per_segment.jsonl.gz ${wenet_misp2025_data_path}/${dset}_far_audio_segment/cuts_per_segment.jsonl.gz
    cp ${Manifest_root}/${dset}_wave/far/wpe/gss_new/recordings.jsonl.gz ${wenet_misp2025_data_path}/${dset}_far_audio_segment/recordings.jsonl.gz
    cp ${Manifest_root}/${dset}_wave/far/wpe/gss_new/supervisions.jsonl.gz ${wenet_misp2025_data_path}/${dset}_far_audio_segment/supervisions.jsonl.gz

    python ${gss_path}/gss2lhotse.py -i $EXP_DIR -o $EXP_DIR/${manifest_file}
    lhotse kaldi export $EXP_DIR/${manifest_file}_recordings.jsonl.gz $EXP_DIR/${manifest_file}_supervisions.jsonl.gz dump/raw/${manifest_file}
    ./utils/utt2spk_to_spk2utt.pl dump/raw/${manifest_file}/utt2spk > dump/raw/${manifest_file}/spk2utt
    ./utils/fix_data_dir.sh dump/raw/${manifest_file}
    cp dump/raw/${manifest_file}/text ${wenet_misp2025_data_path}/${dset}_far_audio_segment/text
    cp dump/raw/${manifest_file}/wav.scp ${wenet_misp2025_data_path}/${dset}_far_audio_segment/wav.scp
fi