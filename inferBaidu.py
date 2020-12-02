"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import argparse
import functools
import paddle.fluid as fluid
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from model_utils.model_check import check_cuda, check_version
from utils.error_rate import wer, cer
from utils.utility import add_arguments, print_arguments
import create_manifest 
import json
import codecs
import soundfile
import time
import numpy as np

ds2_model = None
data_generator = None
vocab_list = None

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('num_samples',      int,    128,     "# of samples to infer.")
add_arg('beam_size',        int,    500,    "Beam search width.")
add_arg('num_proc_bsearch', int,    8,      "# of CPUs for beam search.")
add_arg('num_conv_layers',  int,    2,      "# of convolution layers.")
add_arg('num_rnn_layers',   int,    3,      "# of recurrent layers.")
add_arg('rnn_layer_size',   int,    1024,   "# of recurrent cells per layer.")                                                                                      
add_arg('alpha',            float,  2.5,    "Coef of LM for beam search.")
add_arg('beta',             float,  0.3,    "Coef of WC for beam search.")
add_arg('cutoff_prob',      float,  1.0,    "Cutoff probability for pruning.")
add_arg('cutoff_top_n',     int,    40,     "Cutoff number for pruning.")
add_arg('use_gru',          bool,   True,  "Use GRUs instead of simple RNNs.")
add_arg('use_gpu',          bool,   False,   "Use GPU or not.")
add_arg('share_rnn_weights',bool,   False,   "Share input-hidden weights across "
                                            "bi-directional RNNs. Not for GRU.")
add_arg('target_dir',   str,
        '/content/DeepSpeech2/DeepSpeech/dataset/librispeech/test-clean/LibriSpeech/test-clean/',
        "Filepath of voice sample testing folder.")
add_arg('infer_manifest',   str,
        'data/manifest.test-clean',
        "Filepath of manifest to infer.")
add_arg('mean_std_path',    str,
        'models/baidu_en8k/mean_std.npz',
        "Filepath of normalizer's mean & std.")
add_arg('vocab_path',       str,
        'models/baidu_en8k/vocab.txt',
        "Filepath of vocabulary.")
add_arg('lang_model_path',  str,
        'models/lm/common_crawl_00.prune01111.trie.klm',
        "Filepath for language model.")
add_arg('model_path',       str,
        'models/baidu_en8k',
        "If None, the training starts from scratch, "
        "otherwise, it resumes from the pre-trained model.")
add_arg('decoding_method',  str,
        'ctc_beam_search',
        "Decoding method. Options: ctc_beam_search, ctc_greedy",
        choices = ['ctc_beam_search', 'ctc_greedy'])
add_arg('error_rate_type',  str,
        'wer',
        "Error rate type for evaluation.",
        choices=['wer', 'cer'])
add_arg('specgram_type',    str,
        'linear',
        "Audio feature type. Options: linear, mfcc.",
        choices=['linear', 'mfcc'])
add_arg('audio_path',    str,
        '',
        "Audio path to test",
        choices=['linear', 'mfcc'])
# yapf: disable
args = parser.parse_args()

def prepare_manifest():
    print("Preparing Manifest")
    create_manifest.prepare_dataset(target_dir=args.target_dir, manifest_path=args.infer_manifest)


def load_model():        
    # check if set use_gpu=True in paddlepaddle cpu version
    check_cuda(args.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    if args.use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    
    # Load model
    data_generator = DataGenerator(
        vocab_filepath=args.vocab_path,
        mean_std_filepath=args.mean_std_path,
        augmentation_config='{}',
        specgram_type=args.specgram_type,
        keep_transcription_text=True,
        place = place,
        is_training = False)
    
    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=args.num_conv_layers,
        num_rnn_layers=args.num_rnn_layers,
        rnn_layer_size=args.rnn_layer_size,
        use_gru=args.use_gru,
        share_rnn_weights=args.share_rnn_weights,
        place=place,
        init_from_pretrained_model=args.model_path)

    # decoders only accept string encoded in utf-8
    vocab_list = data_generator.vocab_list

    ds2_model.init_ext_scorer(args.alpha, args.beta, args.lang_model_path,
                              vocab_list)
            
    return ds2_model, data_generator, vocab_list
    
    
def infer(ds2_model, data_generator, vocab_list):
    """Inference for DeepSpeech2."""
    
    # Prepare manifest
    if args.audio_path:
        json_lines = []
        audio_data, samplerate = soundfile.read(args.audio_path)
        duration = float(len(audio_data)) / samplerate
        json_lines.append(
                    json.dumps({
                        'audio_filepath': args.audio_path,
                        'duration': duration,
                        'text': 'NO TRANSCRIPT'
                    }))
        with codecs.open(args.infer_manifest, 'w', 'utf-8') as out_file:
            for line in json_lines:
                 out_file.write(line + '\n')
    else:
        prepare_manifest()
        
    # Load audio
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=args.infer_manifest,
        batch_size=args.num_samples,
        sortagrad=False,
        shuffle_method=None)
    #infer_data = next(batch_reader()) # (padded_audios, texts, audio_lens, masks, audio_file_path)
    error_rate_func = cer if args.error_rate_type == 'cer' else wer
    errors_sum, len_refs, num_ins = 0.0, 0, 0
    error_arr = []
    ds2_model.logger.info("\nEverything Prepared .. Starting inference ...\n")
    for infer_data in batch_reader():
        probs_split= ds2_model.infer_batch_probs(
            infer_data=infer_data,
            feeding_dict=data_generator.feeding)
        result_transcripts= ds2_model.decode_batch_beam_search(
            probs_split=probs_split,
            beam_alpha=args.alpha,
            beam_beta=args.beta,
            beam_size=args.beam_size,
            cutoff_prob=args.cutoff_prob,
            cutoff_top_n=args.cutoff_top_n,
            vocab_list=vocab_list,
            num_processes=args.num_proc_bsearch)
        target_transcripts = infer_data[1]
        audio_file_paths = infer_data[4]
        json_lines = []
        print("Writing Results on TRANSCRIPTION.json ...")
        for target, result, audio_file_path  in zip(target_transcripts, result_transcripts,audio_file_paths):
            target = target.replace("’", "'")
            erroris = error_rate_func(target, result)
            json_lines.append(
                        json.dumps({
                            'Audio file path': audio_file_path,
                            'Target Transcription': target,
                            'Output Transcription': result,
                            'The {} '.format(args.error_rate_type): erroris,
                        }, indent=4, ensure_ascii=False, sort_keys=True))
            error_arr.append(erroris)
        wer_arr = np.array(error_arr)
        print("Current Error Rate is : ",str(np.average(wer_arr)))
        with codecs.open('TRANSCRIPTION.json', 'a+', 'utf-8') as out_file:
            for line in json_lines:
                out_file.write(line + '\n')
    with codecs.open('TRANSCRIPTION.json', 'a+', 'utf-8') as out_file:
            out_file.write("Average Error Rate is : " + str(np.average(wer_arr)) +'\n')
    ds2_model.logger.info("Finished Inference.")
    if args.audio_path:
        return result_transcripts.pop()

def main():
    global ds2_model
    global data_generator
    global vocab_list
    print_arguments(args)
    #args.audio_path = audio_path
    
    if not ds2_model:
        print("\nModel Loading Initiated ...")        
        ds2_model, data_generator, vocab_list = load_model()  
        print("\nModel Loaded Successfully ...\n")        
        tic = time.time()
        result_transcripts = infer(ds2_model, data_generator, vocab_list)
        toc = time.time()
        print("{} Mins Required For Transcription".format(toc-tic)/60)
    else:
        tic = time.time()
        result_transcripts = infer(ds2_model, data_generator, vocab_list)
        toc = time.time()
        print("{} Mins Required For Transcription".format(toc-tic))
    return result_transcripts

if __name__ == '__main__':
    print(main())
