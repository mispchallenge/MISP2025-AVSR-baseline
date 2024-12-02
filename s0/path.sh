export WENET_DIR=$PWD/../../..
export BUILD_DIR=${WENET_DIR}/runtime/libtorch/build
export OPENFST_BIN=${BUILD_DIR}/../fc_base/openfst-build/src
export PATH=$PWD:${BUILD_DIR}/bin:${BUILD_DIR}/kaldi:${OPENFST_BIN}/bin:$PATH

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../../:$PYTHONPATH

# 导入anaconda环境
export PATH=/home4/intern/zhewang18/anaconda3/envs/wenet2/bin:${PATH}
# 导入gcc环境
export PATH=/opt/compiler/gcc-7.3.0-os7.2/bin:$PATH
export LD_LIBRARY_PATH=/opt/compiler/gcc-7.3.0-os7.2/lib64:$LD_LIBRARY_PATH
# 导入perl环境
export PATH=/home3/cv1/hangchen2/localperl/bin:${PATH}
# 导入delta库
# export PATH=/home3/cv1/hangchen2/delta-master/delta/bin:${PATH}

# 单机多卡NCCL报错no socket interface found,手动设置NCCL_SOCKET_IFNAME=eth0。多机多卡NCCL_SOCKET_IFNAME=eno2.100
export NCCL_SOCKET_IFNAME=eth0
# ifconfig查看ip配置调试机是eno1:172.20.101.131/172.20.98.196
# export NCCL_SOCKET_IFNAME=eno1
