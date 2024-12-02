#!/usr/bin/env bash
# Copyright 2018 USTC (Authors: Zhaoxu Nian, Hang Chen)
# Apache 2.0

# use nara-wpe and GSS to enhance multichannel data

set -eo pipefail
# configs
stage=0
nj=10
python_path=
train_cmd=

. ./path.sh || exit 1
. tools/parse_options.sh || exit 1;
if [ $# != 2 ]; then
 echo "Usage: $0 <corpus-data-dir> <enhancement-dir>"
 echo " $0 /path/misp2021 /path/gss_output"
 exit 1;
fi

data_root=$1
out_root=$2

echo "start speech enhancement"

# GSS

if [ $stage -le 1 ]; then
  echo "start gss"
  mkdir -p $out_root/log
  mkdir -p $out_root/wav
  ${python_path}python local/find_wav.py ${data_root} $out_root/log gss Far -nj $nj
  for n in `seq $nj`; do
    cat <<-EOF > $out_root/log/gss.$n.sh
    ${python_path}python local/run_gss.py $out_root/log/gss.$n.scp ${data_root} $out_root Far
EOF
  done
  chmod a+x $out_root/log/gss.*.sh
  $train_cmd JOB=1:$nj $out_root/log/gss.JOB.log $out_root/log/gss.JOB.sh
  echo "finish gss"
fi