#! /usr/bin/env bash

. ../../utils/utility.sh

URL='https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm'
MD5="29e02312deb2e59b3c8686c7966d4fe3"
TARGET=./zh_giga.no_cna_cmn.prune01244.klm

echo "Download language model ..."
download $URL $MD5 $TARGET
if [ $? -ne 0 ]; then
    echo "Fail to download the language model!"
    exit 1
fi
exit 0