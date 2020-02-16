"""Script to compute question embeddings for given questions."""
import cPickle as pkl
import numpy as np
import json
from tqdm import tqdm
import tensorflow as tf
import gzip
from fat.fat_bert_nq import nq_data_utils

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('nq_dir', '/remote/bones/user/vbalacha/datasets/ent_linked_nq_new/', 'Read nq data to extract entities')
flags.DEFINE_string('output_dir', '/remote/bones/user/vbalacha/datasets/nq_question_embeddings/', 'Read nq data to extract entities')
flags.DEFINE_integer("shard_id", None,
                     "Train and dev shard to read from and write to.")
flags.DEFINE_integer("task_id", None,
                     "Train and dev shard to read from and write to.")
flags.DEFINE_string(
    "model", "train",
    "Train and dev split to read from and write to. Accepted values: ['train', 'dev', 'test']"
)

questions_file = nq_data_utils.get_sharded_filename(FLAGS.nq_dir, FLAGS.mode, FLAGS.task_id, FLAGS.shard_id, 'jsonl.gz')
embeddings_file = "/remote/bones/user/vbalacha/datasets/glove/glove.6B.300d.txt"
output_file = nq_data_utils.get_sharded_filename(FLAGS.nq_dir, FLAGS.mode, FLAGS.task_id, FLAGS.shard_id, 'pkl')
dim = 300

word_to_question = {}
question_lens = {}

def _add_word(word, v):
    if word not in word_to_question: word_to_question[word] = []
    word_to_question[word].append(v)
    if v not in question_lens: question_lens[v] = 0
    question_lens[v] += 1

with gzip.GzipFile(fileobj=tf.gfile.Open(questions_file, "rb")) as input_file:
    for line in input_file:
        data = json.loads(line)
        qId, question_text = data["example_id"], data["question_text"]
        for word in question_text.split():
            _add_word(word, qId)

question_emb = {r: np.zeros((dim,)) for r in question_lens}
with open(embeddings_file) as f:
    for line in tqdm(f):
        word, vec = line.strip().split(None, 1)
        if word in word_to_question:
            for qid in word_to_question[word]:
                question_emb[qid] += np.array([float(vv) for vv in vec.split()])

for question in question_emb:
    question_emb[question] = question_emb[question] / question_lens[question]

pkl.dump(question_emb, open(output_file, "w"))