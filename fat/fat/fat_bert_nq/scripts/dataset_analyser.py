import collections
import gzip
import json
import os
import random
import re

import enum

from bert import modeling
from bert import optimization
from bert import tokenization

import numpy as np
import tensorflow as tf

def get_inp_data(record_iterator):
    data = []
    counter = 0
    fact_counter = []
    d = []
    for string_record in record_iterator:
        counter += 1
        if counter == 1000:
            break
        example = tf.train.Example()
        example.ParseFromString(string_record)
        input_ids = list(example.features.feature['input_ids'].int64_list.value)
        input_ids = map(int, input_ids)

        words = []
        c=0
        f=False
        question=[]
        q=False
        passage = []
        p = False
        facts= []
        fac = False

        for token in input_ids:
            try:
                word = tokenizer.convert_ids_to_tokens([token])[0]
                if word == '[CLS]':
                    q=True
                    continue
                if word == '[SEP]' and q:
                    q=False
                    p=True
                    continue
                if word=='[SEP]' and p:
                    q=False
                    p=False
                    fac=True
                    f=True
                if q:
                    question.append(word)
                if p:
                    passage.append(word)
                if fac:
                    facts.append(word)

                if f and word=='.':
                    c+=1
                    words.append(word)
            except:
                f=True
        data.append((question, passage, facts))
    return data

vocab_path = '/remote/bones/user/vbalacha/bert-joint-baseline/vocab-nq.txt'
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_path, do_lower_case=True)

old_file_path = '/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/data_mc512_unk0.02_test/train/nq-train-0000.tf-record'
record_iterator = tf.python_io.tf_record_iterator(path=old_file_path)
old_data = get_inp_data(record_iterator)
with open('old_data.txt', 'w') as fp:
    for (q, p, f) in old_data:
        fp.write('Question: \n'+" ".join(q)+"\n")
        fp.write('Passage: \n'+" ".join(p)+"\n")
        fp.write('Facts: \n'+" ".join(f)+"\n\n")

new_file_path = '/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/sharded_kb_data_mc512_unk0.02_test/train/nq-train-0000.tf-record'
record_iterator2 = tf.python_io.tf_record_iterator(path=new_file_path)
new_data = get_inp_data(record_iterator2)
with open('new_data.txt', 'w') as fp:
    for (q, p, f) in new_data:
        fp.write('Question: \n'+" ".join(q)+"\n")
        fp.write('Passage: \n'+" ".join(p)+"\n")
        fp.write('Facts: \n'+" ".join(f)+"\n\n")
