# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file does create, save and load of the KB graph in Scipy CSR format."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gc
import gzip
import json
import os
import tempfile

import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
import tensorflow as tf

from fat.fat_bert_nq import nq_data_utils
from fat.fat_bert_nq.ppr import sling_utils
from fat.fat_bert_nq.ppr.apr_lib import ApproximatePageRank
from fat.fat_bert_nq.ppr.kb_csr_io import CsrData


flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('nq_dir', 'None', 'Read nq data to extract entities')
flags.DEFINE_string('apr_files_dir', 'None', 'Read and Write apr data')
flags.DEFINE_bool('full_wiki', True, '')
flags.DEFINE_bool('decompose_ppv', False, '')
flags.DEFINE_integer('total_kb_entities', 29964483,
                     'Total entities in processed sling KB')
#188309 for sub graphs
#29964483 for full graphs


def extract_nq_data(nq_file):
    """Read nq shard file and return dict of nq_data."""
    fp = gzip.GzipFile(fileobj=tf.gfile.Open(nq_file, "rb"))
    lines = fp.readlines()
    data = {}
    entities = []
    counter = 0
    for line in lines:
        item = json.loads(line.decode("utf-8"))
        data[str(counter)] = item
        entities.extend([ ent for k, v in item['question_entity_map'] for (id, ent) in v ])
        for ann in item["annotations"]:
            entities.extend([ ent for k, v in ann["long_answer"]["entity_map"] for (id, ent) in v ])
        for cand in item["long_answer_candidates"]:
            entities.extend([ ent for k, v in cand["entity_map"] for (id, ent) in v ])
        for ann in item["annotations"]:
            for sa in ann['short_answers']:
                entities.extend([ ent for k, v in sa["entity_map"] for (id, ent) in v ])
        counter += 1
    return data, list(set(entities))


# def get_shard(mode, task_id, shard_id):
#     return "nq-%s-%02d%02d" % (mode, task_id, shard_id)
#
#
# def get_full_filename(data_dir, mode, task_id, shard_id):
#     return os.path.join(
#         data_dir, "%s/%s.jsonl.gz" % (mode, get_shard(mode, task_id, shard_id)))


def get_examples(data_dir, mode, task_id, shard_id):
    """Reads NQ data, does sling entity linking and returns augmented data."""
    file_path = nq_data_utils.get_sharded_filename(data_dir, mode, task_id, shard_id, 'jsonl.gz')
    tf.logging.info("Reading file: %s" % (file_path))
    if not os.path.exists(file_path):
        return None
    nq_data, entities = extract_nq_data(file_path)
    tf.logging.info("NQ data Size: " + str(len(nq_data.keys())))
    return nq_data, entities

if __name__ == '__main__':
    print(FLAGS.full_wiki)
    print(FLAGS.decompose_ppv)
    print(FLAGS.apr_files_dir)
    max_tasks = {"train": 50, "dev": 5}
    max_shards = {"train": 6, "dev": 16}
    apr = ApproximatePageRank()
    for mode in ["train"]:
        # Parse all shards in each mode
        # Currently sequentially, can be parallelized later
        for task_id in range(0, max_tasks[mode]):
            for shard_id in range(0, max_shards[mode]):
                nq_data, entities = get_examples(FLAGS.nq_dir, mode, task_id, shard_id)
                if nq_data is None:
                    continue
                two_hop_entities = apr.get_khop_entities(entities, 3)
                print("Size of two hop entities: %d", two_hop_entities)
                csr_data = CsrData()
                csr_data.create_and_save_csr_data(full_wiki=FLAGS.full_wiki,
                                                  decompose_ppv=FLAGS.decompose_ppv,
                                                  files_dir=FLAGS.apr_files_dir,sub_entities=two_hop_entities, mode=mode, task_id=task_id, shard_id=shard_id)


