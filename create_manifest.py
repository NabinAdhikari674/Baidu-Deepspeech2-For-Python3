"""Prepare Librispeech ASR datasets.

Download, unpack and create manifest files.
Manifest file is a json-format file with each line containing the
meta data (i.e. audio filepath, transcript and audio duration)
of each audio file in the data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import distutils.util
import os
import sys
import argparse
import soundfile
import json
import codecs
import io
# sudo pip install paddlepaddle
from paddle.dataset.common import md5file
import tarfile
# from data_utils.utility import download, unpack
import sys

def create_manifest(data_dir, manifest_path):
    """Create a manifest json file summarizing the data set, with each line
    containing the meta data (i.e. audio filepath, transcription text, audio
    duration) of each audio file within the data set.
    """
    print("Creating manifest %s ..." % manifest_path)
    json_lines = []
    for subfolder, _, filelist in sorted(os.walk(data_dir)):
        text_filelist = [
            filename for filename in filelist if filename.endswith('trans.txt')
        ]
        if len(text_filelist) > 0:
            text_filepath = os.path.join(subfolder, text_filelist[0])
            for line in io.open(text_filepath, encoding="utf8"):
                segments = line.strip().split()
                text = ' '.join(segments[1:]).lower()
                audio_filepath = os.path.join(subfolder, segments[0] + '.wav')
                audio_data, samplerate = soundfile.read(audio_filepath)
                duration = float(len(audio_data)) / samplerate
                json_lines.append(
                    json.dumps({
                        'audio_filepath': audio_filepath,
                        'duration': duration,
                        'text': text
                    }))
    with codecs.open(manifest_path, 'w', 'utf-8') as out_file:
        for line in json_lines:
            out_file.write(line + '\n')


def prepare_dataset(target_dir, manifest_path):
    """Download, unpack and create summmary manifest file.
    """
    # create manifest json file
    create_manifest(target_dir, manifest_path)
    
if __name__ == '__main__':
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        manifest_path = sys.argv[2]
        prepare_dataset(target_dir, manifest_path)
    
    
    
    
