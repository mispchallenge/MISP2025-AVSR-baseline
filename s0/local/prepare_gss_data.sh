#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Dalision)
# Apache 2.0

# transform misp data to kaldi format

set -e -o pipefail
echo "$0 $@"
nj=1
need_channel=true
stage=1

gss_path=$1
gss_dir=$2
channel_dir=${gss_dir}
transcription_dir=$3
data_type=$4
store_dir=$5
video_dir=None


# wav.scp segments text_sentence utt2spk
# you can change --without_mp4,enhancement_wav,video_path to combine different audio and video filed as you like

if [ ${stage} -le 1 ];then
    _opt=""
    echo "prepare wav.scp segments text_sentence utt2spk"
    echo "$need_channel"
    [[ -n $need_channel ]] && _opts+="--channel_dir $channel_dir --without_wav False"
    python ${gss_path}/prepare_gss_data.py $_opts --without_wav true $gss_dir $video_dir $transcription_dir $data_type $store_dir
fi

#fix kaldi data dir
if [ ${stage} -le 2 ];then
    for file in wav.scp channels.scp mp4.scp segments utt2spk text_sentence;do
        if [ -f $store_dir/temp/$file ];then
            if [ $file == "text_sentence" ];then 
                cat $store_dir/temp/$file | sort -k 1 | uniq > $store_dir/text 
            else
                cat $store_dir/temp/$file | sort -k 1 | uniq > $store_dir/$file
            fi
        fi
    done
    
    [[ -e $store_dir/temp/channels.scp ]] && cp $store_dir/channels.scp $store_dir/wav.scp
    # rm -r $store_dir/temp
    echo "prepare done"

    # generate spk2utt and nlsyms
    cd ${gss_path}
    utils/utt2spk_to_spk2utt.pl $store_dir/utt2spk | sort -k 1 | uniq > $store_dir/spk2utt
    touch data/nlsyms.txt
    
    utils/fix_data_dir.sh $store_dir

fi

echo "prepare_gss_data.sh succeeded"
exit 0