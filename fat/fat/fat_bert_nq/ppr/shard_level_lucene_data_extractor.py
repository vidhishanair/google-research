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

flags.DEFINE_string('nq_dir', '/remote/bones/user/vbalacha/datasets/ent_linked_nq_new/', 'Read nq data to extract entities')
flags.DEFINE_integer("shard_split_id", None,
                     "Train and dev shard to read from and write to.")

flags.DEFINE_string(
    "split", "train",
    "Train and dev split to read from and write to. Accepted values: ['train', 'dev', 'test']"
)
#flags.DEFINE_integer("task_id", 0,
#                             "Train and dev shard to read from and write to.")
flags.DEFINE_string('output_data_dir', 'None', 'Read and Write apr data')
#flags.DEFINE_bool('full_wiki', True, '')
#flags.DEFINE_bool('decompose_ppv', False, '')
#flags.DEFINE_integer('total_kb_entities', 29964483,
#                     'Total entities in processed sling KB')
#188309 for sub graphs
#29964483 for full graphs


def extract_nq_data(nq_file, output_file):
    """Read nq shard file and return dict of nq_data."""
    fp = gzip.GzipFile(fileobj=tf.gfile.Open(nq_file, "rb"))
    op = tf.gfile.Open(output_file, 'w')
    lines = fp.readlines()
    data = {}
    entities = []
    counter = 0
    for line in lines:
        item = json.loads(line.decode("utf-8"))
        data[str(counter)] = item
        doc_tokens = [token['token'] for token in item['document_tokens']]
        gold_short_answers = []
        for gold_label in item['annotations']:
            for g in gold_label['short_answers']:
              start_tok = g['start_token']
              end_tok = g['end_token']
              g_answer = doc_tokens[start_tok:end_tok]
              gold_short_answers.append(" ".join(g_answer))

        short_answer = " | ".join(gold_short_answers)
        op.write(str(item['example_id'])+"\t"+str(item['question_text'])+"\t"+short_answer+"\n")
        if 'question_entity_map' in item.keys():
            entities.extend([ ent for k, v in item['question_entity_map'].items() for (ids, ent) in v ])
        for ann in item["annotations"]:
            if 'entity_map' in ann['long_answer'].keys():
                entities.extend([ ent for k, v in ann["long_answer"]["entity_map"].items() for (ids, ent) in v ])
        for cand in item["long_answer_candidates"]:
            if 'entity_map' in cand.keys():
                entities.extend([ ent for k, v in cand["entity_map"].items() for (ids, ent) in v ])
        for ann in item["annotations"]:
            for sa in ann['short_answers']:
                if 'entity_map' in sa.keys():
                    entities.extend([ ent for k, v in sa["entity_map"].items() for (ids, ent) in v ])
        counter += 1
    return data, list(set(entities))


# def get_shard(mode, task_id, shard_id):
#     return "nq-%s-%02d%02d" % (mode, task_id, shard_id)
#
#
# def get_full_filename(data_dir, mode, task_id, shard_id):
#     return os.path.join(
#         data_dir, "%s/%s.jsonl.gz" % (mode, get_shard(mode, task_id, shard_id)))


def get_examples(data_dir, output_data_dir, mode, task_id, shard_id):
    """Reads NQ data, does sling entity linking and returns augmented data."""
    file_path = nq_data_utils.get_sharded_filename(data_dir, mode, task_id, shard_id, 'jsonl.gz')
    output_file_path = nq_data_utils.get_sharded_filename(output_data_dir, mode, task_id, shard_id, 'tsv')
    print(file_path)
    tf.logging.info("Reading file: %s" % (file_path))
    if not os.path.exists(file_path):
        return None, None
    nq_data, entities = extract_nq_data(file_path, output_file_path)
    tf.logging.info("NQ data Size: " + str(len(nq_data.keys())))
    return nq_data, entities

def get_file_names(full_wiki, files_dir, output_dir, shard_level=False, mode=None, task_id=None, shard_id=None):
    """Return filenames depending on full KB or subset of KB."""
    if shard_level:
        file_names = {
            # 'ent2id_fname': 'csr_ent2id_full.json.gz',
            # 'id2ent_fname': 'csr_id2ent_full.json.gz',
            # 'rel2id_fname': 'csr_rel2id_full.json.gz',
            # 'rel_dict_fname': 'csr_rel_dict_full.npz',
            # 'entity_names_fname': 'csr_entity_names_full.json.gz',
            # 'adj_mat_fname': 'csr_adj_mat_sparse_matrix_full.npz',
            'facts_fname': 'facts_full.tsv'
        }
        sharded_fnames = {k: '%02d%02d_'%(task_id, shard_id) + v for k, v in file_names.items()}
        file_paths = {k: os.path.join(output_dir+"%s/"%(mode), v) for k, v in sharded_fnames.items()}
        file_paths['kb_fname'] =  os.path.join(files_dir, 'kb.sling')
    else:
        sub_file_names = {
#             'ent2id_fname': 'csr_ent2id_sub.json.gz',
#             'id2ent_fname': 'csr_id2ent_sub.json.gz',
#             'rel2id_fname': 'csr_rel2id_sub.json.gz',
#             'rel_dict_fname': 'csr_rel_dict_sub.npz',
#             'kb_fname': 'kb.sling',
#             'entity_names_fname': 'csr_entity_names_sub.json.gz',
#             'adj_mat_fname': 'csr_adj_mat_sparse_matrix_sub.npz',
            'facts_fname': 'facts_sub.tsv'
        }
        full_file_names = {
#             'ent2id_fname': 'csr_ent2id_full.json.gz',
#             'id2ent_fname': 'csr_id2ent_full.json.gz',
#             'rel2id_fname': 'csr_rel2id_full.json.gz',
#             'rel_dict_fname': 'csr_rel_dict_full.npz',
#             'kb_fname': 'kb.sling',
#             'entity_names_fname': 'csr_entity_names_full.json.gz',
#             'adj_mat_fname': 'csr_adj_mat_sparse_matrix_full.npz',
            'facts_fname': 'facts_full.tsv'
        }
        files = full_file_names if full_wiki else sub_file_names
        file_paths = {k: os.path.join(files_dir, v) for k, v in files.items()}
    return file_paths

def create_and_save_csr_data(full_wiki, decompose_ppv, files_dir,
                             sub_entities=None, mode=None, task_id=None,
                             shard_id=None, output_dir=None):
    """Return the PPR vector for the given seed and adjacency matrix.

      Algorithm : Parses sling KB - extracts subj, obj, rel triple and stores
        as sparse matrix.
      Data Store :
          ent2id = json {'Q123':1}
          rel2id = json {'P123':1}
          entity_names = json { 'e':{ 'Q123':'abc'}, 'r':{ 'P123':'abc'} }
          adj_mat = ExE scipy CSR matrix reldict = ExE scipy DOK matrix
    Args:
      full_wiki : boolean True which Parses entire Wikidata
      decompose_ppv : boolean True which
                  Creates Relation level SP Matrices and then combines them
      files_dir : Directory to save KB data in
      sub_entities : entities to keep for building sharded graph

    Returns:
      None
    """
    if sub_entities is not None and mode is not None and task_id is not None and shard_id is not None:
        shard_level = True
    else:
        shard_level = False

    file_paths = get_file_names(full_wiki, files_dir, output_dir, shard_level, mode, task_id, shard_id)
    tf.logging.info('KB Related filenames: %s'%(file_paths))
    print(file_paths)
    tf.logging.info('Loading KB')
    kb = sling_utils.get_kb(file_paths['kb_fname'])

    op = tf.gfile.Open(file_paths['facts_fname'], 'w')

    sub_entities = {k:1 for k in sub_entities}
    count = 0

    tf.logging.info('Processing KB')
    for x in kb:
        count += 1
        if not full_wiki and count == 100000:
            break  # For small KB Creation
        if sling_utils.is_subj(x, kb):
            subj = x.id
            properties = sling_utils.get_properties(x, kb)
            for (rel, obj) in properties:
                if sub_entities is not None and (subj not in sub_entities or obj not in sub_entities):
                    continue
                fact = str(kb[subj]['name']) + " " + str(kb[rel]['name']) + " " + str(kb[obj]['name'])
                op.write(str(count)+"\t"+str(fact)+"\n")


if __name__ == '__main__':
    print(FLAGS.full_wiki)
    print(FLAGS.decompose_ppv)
    print(FLAGS.apr_files_dir)
    max_tasks = {"train": 50, "dev": 5}
    max_shards = {"train": 7, "dev": 17}
    apr = ApproximatePageRank()
    create_and_save_csr_data(full_wiki=FLAGS.full_wiki,
                                                      decompose_ppv=FLAGS.decompose_ppv,
                                                      files_dir=FLAGS.apr_files_dir,
                                                      output_dir=FLAGS.output_data_dir)
#     for mode in [FLAGS.split]:
#         # Parse all shards in each mode
#         # Currently sequentially, can be parallelized later
#         for task_id in [FLAGS.task_id]: #range(0, max_tasks[mode]):
#             for shard_id in [FLAGS.shard_split_id]: #range(0, max_shards[mode]):
#                 # if task_id == 0 and shard_id in range(0, 16):
#                 #     print("skipping finished job")
#                 #     continue
#                 nq_data, entities = get_examples(FLAGS.nq_dir, FLAGS.output_data_dir, mode, task_id, shard_id)
#                 if nq_data is None:
#                     print("No examples here")
#                     continue
#                 print("Size of all entities: %d", len(entities))
#                 two_hop_entities = apr.get_khop_entities(entities, 2)
#                 print("Size of two hop entities: %d", len(two_hop_entities))
#                 # csr_data = CsrData()
#                 create_and_save_csr_data(full_wiki=FLAGS.full_wiki,
#                                                   decompose_ppv=FLAGS.decompose_ppv,
#                                                   files_dir=FLAGS.apr_files_dir,
#                                                   sub_entities=two_hop_entities,
#                                                   mode=mode,
#                                                   task_id=task_id,
#                                                   shard_id=shard_id,
#                                                   output_dir=FLAGS.output_data_dir)

