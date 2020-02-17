"""Script to compute relation embeddings for given relation."""
#import cPickle as pkl
import pickle as pkl
import numpy as np
import json
from tqdm import tqdm
import tensorflow as tf
import gzip
from fat.fat_bert_nq import nq_data_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('relations_file', None, 'input relations dict')
flags.DEFINE_string('rel2id_file', None, 'input relations dict')
flags.DEFINE_string('output_file', None, 'output relations dict')

relations_file = FLAGS.relations_file
embeddings_file = "/remote/bones/user/vbalacha/datasets/glove/glove.6B.300d.txt"
output_file = FLAGS.output_file
dim = 300

word_to_relation = {}
relation_lens = {}

with gzip.GzipFile(fileobj=tf.gfile.Open(FLAGS.rel2id_file, 'rb')) as op4:
    rel2id = json.load(op4)
    op4.close()
id2rel = {str(idx): ent for ent, idx in rel2id.items()}

def _add_word(word, v):
    if word not in word_to_relation: word_to_relation[word] = []
    word_to_relation[word].append(v)
    if v not in relation_lens: relation_lens[v] = 0
    relation_lens[v] += 1

rel_dict = {}
with gzip.GzipFile(fileobj=tf.gfile.Open(relations_file, 'rb')) as op4:
    obj = json.load(op4)
    op4.close()
    relations = obj['r']
    for rel_id, val in relations.items():
        rel_name = val['name']
        rel = id2rel[rel_id]
        rel_dict[rel] = rel_name
        for word in rel_name.split():
            _add_word(word.lower(), rel)
        if rel_name == 'director/manager':
            for word in rel_name.split('/'):
                _add_word(word.lower(), rel)
        if rel_name == 'vice-county' or 'Sandbox-Item':
            for word in rel_name.split('-'):
                _add_word(word.lower(), rel)


relation_emb = {r: np.zeros((dim,)) for r in relation_lens}
with open(embeddings_file) as f:
    for line in tqdm(f):
        word, vec = line.strip().split(None, 1)
        if word in word_to_relation:
            for qid in word_to_relation[word]:
                relation_emb[qid] += np.array([float(vv) for vv in vec.split()])

pruned_relation_emb = {}
for relation in relation_emb:
    if np.count_nonzero(relation_emb[relation]) == 0:
        print(rel_dict[relation])
        relation_emb[relation] = np.random.rand(300)
        continue
    pruned_relation_emb[relation] = relation_emb[relation] / relation_lens[relation]
#print(relation_emb)
print("Processed relations: " + str(len(pruned_relation_emb.keys())))
pkl.dump(pruned_relation_emb, open(output_file, "wb"))
