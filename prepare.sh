PYTHONPATH=.:$PYTHONPATH python data/librispeech/librispeech.py \
--manifest_prefix='data/librispeech/manifest' \
--target_dir='./dataset/librispeech' \
--full_download='False'
if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi
echo "LibriSpeech Data preparation done."

cd models/librispeech
sh download_model.sh
cd ../baidu_en8k
sh download_model.sh
echo "Pre-Trained Models[Librispeech and Baidu_en8k] Downloaded."
cd ../lm
download_lm_en.sh
echo "English Language Model Downloaded"
 
cd ../..

echo " Data And Model Preparation Done. Finished All Preparations."
exit 0