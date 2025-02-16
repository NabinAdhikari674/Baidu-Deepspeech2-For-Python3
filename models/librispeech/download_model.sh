#! /usr/bin/env bash

. ../../utils/utility.sh

URL='https://deepspeech.bj.bcebos.com/eng_models/librispeech_model_fluid.tar.gz'
MD5=fafb11fe57c3ecd107147056453f5348
TARGET=./librispeech_model_fluid.tar.gz

echo "Download LibriSpeech model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download LibriSpeech model!"
    exit 1
fi
tar -zxvf $TARGET
exit 0