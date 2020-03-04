#!/usr/bin/env python3
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

"""BERT-joint baseline for NQ v1.0."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import gzip
import json
import os
import random
import re
import time 

import enum

from bert import modeling
from bert import optimization
from bert import tokenization

import numpy as np
import tensorflow as tf

import spacy

from fat.fat_bert_nq.ppr.shortest_path_lib import ShortestPath

nlp = spacy.load("en_core_web_lg")

from fat.fat_bert_nq.ppr.apr_lib import ApproximatePageRank

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("eval_data_path", None, "Precomputed eval path for dev set")

flags.DEFINE_string("train_precomputed_file", None,
                    "Precomputed tf records for training.")

flags.DEFINE_integer("train_num_precomputed", None,
                     "Number of precomputed tf records for training.")

# flags.DEFINE_string("apr_files_dir", None,
#                      "Location of KB for Approximate Page Rank")

flags.DEFINE_string(
    "predict_file", None,
    "NQ json for predictions. E.g., dev-v1.1.jsonl.gz or test-v1.1.jsonl.gz")

flags.DEFINE_string(
    "output_prediction_file", None,
    "Where to print predictions in NQ prediction format, to be passed to"
    "natural_questions.nq_eval.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 384,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "doc_stride", 128,
    "When splitting up a long document into chunks, how much stride to "
    "take between chunks.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")

flags.DEFINE_bool("create_pretrain_data", False, "Whether to create_pretraining_data.")

flags.DEFINE_bool(
    "use_random_fact_generator", False,
    "Whether to retreive random facts "
    "models and False for cased models.")

# flags.DEFINE_bool(
#     "use_question_entities", False,
#     "Whether to use question entities as seeds "
#     "models and False for cased models.")
flags.DEFINE_bool(
    "mask_non_entity_in_text", False,
    "Whether to mask non entity tokens "
    "models and False for cased models.")
flags.DEFINE_bool(
    "use_text_only", False,
    "Whether to use text only version of masked non entity tokens "
    "models and False for cased models.")
flags.DEFINE_bool(
    "use_masked_text_only", False,
    "Whether to use text only version of masked non entity tokens "
    "models and False for cased models.")
flags.DEFINE_bool(
    "use_text_and_facts", False,
    "Whether to use text and facts version of masked non entity tokens "
    "models and False for cased models.")
flags.DEFINE_bool(
    "augment_facts", False,
    "Whether to do fact extraction and addition ")
flags.DEFINE_bool(
    "anonymize_entities", False,
    "Whether to do add anonymized version")
flags.DEFINE_bool(
    "use_named_entities_to_filter", False,
    "Whether to use_ner_to_filter")

flags.DEFINE_bool(
    "use_shortest_path_facts", False,
    "Whether to do shortest_path expt")
flags.DEFINE_bool(
    "shuffle_shortest_path_facts", False,
    "Whether to do shuffle hortest_path expt")
flags.DEFINE_bool(
    "add_random_question_facts_to_shortest_path", False,
    "Whether to retreive random facts "
    "models and False for cased models.")
flags.DEFINE_bool(
    "add_random_walk_question_facts_to_shortest_path", False,
    "Whether to retreive random walk facts "
    "models and False for cased models.")
flags.DEFINE_bool(
    "use_passage_rw_facts_in_shortest_path", False,
    "Whether to do shuffle hortest_path expt")
flags.DEFINE_bool(
    "use_question_rw_facts_in_shortest_path", False,
    "Whether to do shuffle hortest_path expt")
flags.DEFINE_bool(
    "create_fact_annotation_data", False,
    "Whether to do shuffle hortest_path expt")
# flags.DEFINE_bool(
#     "use_only_random_facts_of_question", False,
#     "Whether to use only random_facts")
flags.DEFINE_bool(
    "use_question_to_passage_facts_in_shortest_path", False,
    "Whether to use only question to passage facts")
flags.DEFINE_bool(
    "use_question_level_apr_data", False,
    "Whether to use only question to passage facts")
flags.DEFINE_bool(
    "use_fixed_training_data", False,
    "Whether to use fixed unique ids from list")
tf.flags.DEFINE_string(
    "fixed_training_data_filepath", None,
    "Filepath of fixed training data - unique ids")

flags.DEFINE_integer("num_facts_limit", -1,
                     "Limiting number of facts")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("use_entity_markers", False, "Whether to add explicit entity seperators")
flags.DEFINE_float("alpha", 0.9,
                   "Alpha as restart prob for RWR")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
    "n_best_size", 20,
    "The total number of n-best predictions to generate in the "
    "nbest_predictions.json output file.")

flags.DEFINE_integer(
    "max_answer_length", 30,
    "The maximum length of an answer that can be generated. This is needed "
    "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_boolean(
    "skip_nested_contexts", True,
    "Completely ignore context that are not top level nodes in the page.")

flags.DEFINE_integer("task_id", 0,
                     "Train and dev shard to read from and write to.")

flags.DEFINE_integer("max_contexts", 48,
                     "Maximum number of contexts to output for an example.")

flags.DEFINE_integer(
    "max_position", 50,
    "Maximum context position for which to generate special tokens.")

TextSpan = collections.namedtuple("TextSpan", "token_positions text")


class AnswerType(enum.IntEnum):
  """Type of NQ answer."""
  UNKNOWN = 0
  YES = 1
  NO = 2
  SHORT = 3
  LONG = 4


class Answer(collections.namedtuple("Answer", ["type", "text", "offset", "entities"])):
  """Answer record.

  An Answer contains the type of the answer and possibly the text (for
  long) as well as the offset (for extractive).
  """

  def __new__(cls, type_, text=None, offset=None, entities=None):
    return super(Answer, cls).__new__(cls, type_, text, offset, entities)


class NqExample(object):
  """A single training/test example."""

  def __init__(self,
               example_id,
               qas_id,
               questions,
               doc_tokens,
               questions_entity_map=None,
               doc_tokens_map=None,
               entity_list=None,
               ner_entity_list=None,
               answer=None,
               start_position=None,
               end_position=None,
               annotation=None):
    self.example_id = example_id
    self.qas_id = qas_id
    self.questions = questions
    self.doc_tokens = doc_tokens
    self.doc_tokens_map = doc_tokens_map
    self.entity_list = entity_list
    self.ner_entity_list = ner_entity_list
    self.question_entity_map = questions_entity_map
    self.answer = answer
    self.start_position = start_position
    self.end_position = end_position
    self.annotation = annotation


def has_long_answer(a):
  return (a["long_answer"]["start_token"] >= 0 and
          a["long_answer"]["end_token"] >= 0)


def should_skip_context(e, idx):
  if (FLAGS.skip_nested_contexts and
      not e["long_answer_candidates"][idx]["top_level"]):
    return True
  elif not get_candidate_text(e, idx).text.strip():
    # Skip empty contexts.
    return True
  else:
    return False


def get_first_annotation(e):
  """Returns the first short or long answer in the example.

  Args:
    e: (dict) annotated example.

  Returns:
    annotation: (dict) selected annotation
    annotated_idx: (int) index of the first annotated candidate.
    annotated_sa: (tuple) char offset of the start and end token
        of the short answer. The end token is exclusive.
  """
  positive_annotations = sorted(
      [a for a in e["annotations"] if has_long_answer(a)],
      key=lambda a: a["long_answer"]["candidate_index"])

  for a in positive_annotations:
    if a["short_answers"]:
      idx = a["long_answer"]["candidate_index"]
      start_token = a["short_answers"][0]["start_token"]
      end_token = a["short_answers"][-1]["end_token"]

      # entities from the entire SA span (cross check if this makes sense)
      answer_entities = set()
      for sa in a['short_answers']:
          for start_idx in sa['entity_map'].keys():
              for sub_span in sa['entity_map'][start_idx]:
                  answer_entities.add(sub_span[1])

      return a, idx, (token_to_char_offset(e, idx, start_token),
                      token_to_char_offset(e, idx, end_token) - 1), list(answer_entities)

  for a in positive_annotations:
    idx = a["long_answer"]["candidate_index"]
    return a, idx, (-1, -1), []

  return None, -1, (-1, -1), []


def get_text_span(example, span):
  """Returns the text in the example's document in the given token span."""
  token_positions = []
  tokens = []
  for i in range(span["start_token"], span["end_token"]):
    t = example["document_tokens"][i]
    if not t["html_token"]:
      token_positions.append(i)
      token = t["token"].replace(" ", "")
      tokens.append(token)
  return TextSpan(token_positions, " ".join(tokens))


def token_to_char_offset(e, candidate_idx, token_idx):
  """Converts a token index to the char offset within the candidate."""
  c = e["long_answer_candidates"][candidate_idx]
  char_offset = 0
  for i in range(c["start_token"], token_idx):
    t = e["document_tokens"][i]
    if not t["html_token"]:
      token = t["token"].replace(" ", "")
      char_offset += len(token) + 1
  return char_offset


def get_candidate_type(e, idx):
  """Returns the candidate's type: Table, Paragraph, List or Other."""
  c = e["long_answer_candidates"][idx]
  first_token = e["document_tokens"][c["start_token"]]["token"]
  if first_token == "<Table>":
    return "Table"
  elif first_token == "<P>":
    return "Paragraph"
  elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
    return "List"
  elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
    return "Other"
  else:
    tf.logging.warning("Unknoww candidate type found: %s", first_token)
    return "Other"


def add_candidate_types_and_positions(e):
  """Adds type and position info to each candidate in the document."""
  counts = collections.defaultdict(int)
  for idx, c in candidates_iter(e):
    context_type = get_candidate_type(e, idx)
    if counts[context_type] < FLAGS.max_position:
      counts[context_type] += 1
    c["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])


def get_candidate_type_and_position(e, idx):
  """Returns type and position info for the candidate at the given index."""
  if idx == -1:
    return "[NoLongAnswer]"
  else:
    return e["long_answer_candidates"][idx]["type_and_position"]


def get_candidate_text(e, idx):
  """Returns a text representation of the candidate at the given index."""
  # No candidate at this index.
  if idx < 0 or idx >= len(e["long_answer_candidates"]):
    return TextSpan([], "")

  # This returns an actual candidate.
  return get_text_span(e, e["long_answer_candidates"][idx])


def get_candidate_entity_map(e, idx, token_map):
  """Return aligned entitiy list for a given tokenized Long Answer Candidate."""
  if idx < 0 or idx >= len(e["long_answer_candidates"]):
    return []
  entity_list = ["None"] * len(token_map)
  if "entity_map" in e["long_answer_candidates"][idx]:
    for key, value in e["long_answer_candidates"][idx]["entity_map"].items():
      start_token = int(key)
      # temp fix for loose aligning of facts.
      # Due to two tokenizers, the indices aren't matching up
      if start_token >= len(token_map):
        continue

      # To avoid every word token having the entity_id
      # We expt BIO tagging

      # entity_list[start_token] = value[0][
      #     1]
      last_idx = sorted(value, key=lambda x: int(x[0]), reverse=True)[0]
      end_token = int(last_idx[0])
      entity = last_idx[1]
      entity_list[start_token] = 'B-'+entity
      if start_token+1 < len(token_map):
          fixed_end_token = min(end_token, len(token_map))
          entity_list[start_token+1:fixed_end_token] = ['I-'+entity]*(fixed_end_token-start_token-1)
      # for item in value:
      #   end_token = int(item[0])
      #   entity = item[1]
      #   if end_token >= len(token_map): # same temp fix
      #     continue
      #   entity_list[
      #      start_token:end_token] = [entity]*(end_token-start_token) #fix
  assert len(entity_list) == len(token_map)
  return entity_list

def get_candidate_ner_entity_map(e, idx, token_map, text):
    """Return aligned entitiy list for a given tokenized Long Answer Candidate."""
    if idx < 0 or idx >= len(e["long_answer_candidates"]):
        return []
    ner_entity_list = ["None"] * len(token_map)
    char_to_token_map = []
    token_idx = 0
    for token, token_id in zip(text.split(" "), token_map):
        for char in token:
            char_to_token_map.append(token_idx)
        char_to_token_map.append("None")
        token_idx+=1

    doc = nlp(text)
    for ent in doc.ents:
        #print(ent.text)
        start_char = ent.start_char
        end_char = ent.end_char
        start_token_id = char_to_token_map[start_char]
        end_token_id = char_to_token_map[end_char-1]
        if start_token_id == 'None' or end_token_id == 'None':
            print("something wrong")
            print(ent.text, start_char, end_char, start_token_id, end_token_id)
            print(text)
            print(char_to_token_map)
            exit()
        #print(start_token_id, end_token_id)
        ner_entity_list[start_token_id] = 'B-ENT'
        if end_token_id != start_token_id:
            #print(start_token_id+1, end_token_id+1, ner_entity_list[start_token_id+1:end_token_id+1])
            ner_entity_list[start_token_id+1:end_token_id+1] = ['I-ENT']*(end_token_id+1-start_token_id-1)
    #print(text.split(" "))
    #print(ner_entity_list)

    assert len(ner_entity_list) == len(token_map)
    return ner_entity_list


def candidates_iter(e):
  """Yield's the candidates that should not be skipped in an example."""
  for idx, c in enumerate(e["long_answer_candidates"]):
    if should_skip_context(e, idx):
      continue
    yield idx, c


def create_example_from_jsonl(line):
  """Creates an NQ example from a given line of JSON."""
  e = json.loads(line, object_pairs_hook=collections.OrderedDict)
  add_candidate_types_and_positions(e)
  annotation, annotated_idx, annotated_sa, annotated_sa_entities = get_first_annotation(e)
  # annotated_idx: index of the first annotated context, -1 if null.
  # annotated_sa: short answer start and end char offsets, (-1, -1) if null.
  question = {"input_text": e["question_text"], "entity_map": e["question_entity_map"]}
  answer = {
      "candidate_id": annotated_idx,
      "span_text": "",
      "span_start": -1,
      "span_end": -1,
      "sa_entities": [],
      "input_text": "long",
  }

  # Yes/no answers are added in the input text.
  if annotation is not None:
    assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
    if annotation["yes_no_answer"] in ("YES", "NO"):
      answer["input_text"] = annotation["yes_no_answer"].lower()

  # Add a short answer if one was found.
  if annotated_sa != (-1, -1):
    answer["input_text"] = "short"
    span_text = get_candidate_text(e, annotated_idx).text
    answer["span_text"] = span_text[annotated_sa[0]:annotated_sa[1]]
    answer["span_start"] = annotated_sa[0]
    answer["span_end"] = annotated_sa[1]
    answer["sa_entities"] = annotated_sa_entities
    expected_answer_text = get_text_span(
        e, {
            "start_token": annotation["short_answers"][0]["start_token"],
            "end_token": annotation["short_answers"][-1]["end_token"],
        }).text
    assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                         answer["span_text"])

  # Add a long answer if one was found.
  elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
    answer["span_text"] = get_candidate_text(e, annotated_idx).text
    answer["span_start"] = 0
    answer["span_end"] = len(answer["span_text"])

  context_idxs = [-1]
  context_list = [{"id": -1, "type": get_candidate_type_and_position(e, -1)}]
  context_list[-1]["text_map"], context_list[-1]["text"] = (
      get_candidate_text(e, -1))
  for idx, _ in candidates_iter(e):
    context = {"id": idx, "type": get_candidate_type_and_position(e, idx)}
    context["text_map"], context["text"] = get_candidate_text(e, idx)
    context["entity_list"] = get_candidate_entity_map(e, idx,
                                                      context["text_map"])
    context["ner_entity_list"] = get_candidate_ner_entity_map(e, idx,
                                                      context["text_map"], context['text'])
    context_idxs.append(idx)
    context_list.append(context)
    if len(context_list) >= FLAGS.max_contexts:
      break

  # Assemble example.
  example = {
      "name": e["document_title"],
      "id": str(e["example_id"]),
      "questions": [question],
      "answers": [answer],
      "has_correct_context": annotated_idx in context_idxs,
      "annotation": annotation
  }

  single_map = []
  single_context = []
  single_entity_list = []
  single_ner_entity_list = []
  offset = 0
  word_offset = 0
  for context in context_list:
    # id = context["id"]
    single_map.extend([-1, -1])
    single_context.append("[ContextId=%d] %s" %
                          (context["id"], context["type"]))
    offset += len(single_context[-1]) + 1
    word_offset += len(single_context[-1].split(" "))
    single_entity_list.extend(["None"] * len(single_context[-1].split(" ")))
    single_ner_entity_list.extend(["None"] * len(single_context[-1].split(" ")))
    if context["id"] == annotated_idx:
      answer["span_start"] += offset
      answer["span_end"] += offset

    # Many contexts are empty once the HTML tags have been stripped, so we
    # want to skip those.
    if context["text"]:
      single_map.extend(context["text_map"])
      single_entity_list.extend(context["entity_list"])
      single_ner_entity_list.extend(context['ner_entity_list'])
      single_context.append(context["text"])
      offset += len(single_context[-1]) + 1

  example["contexts"] = " ".join(single_context)
  example["contexts_map"] = single_map
  example["context_entity_list"] = single_entity_list
  example["context_ner_entity_list"] = single_ner_entity_list
  if annotated_idx in context_idxs:
    expected = example["contexts"][answer["span_start"]:answer["span_end"]]

    # This is a sanity check to ensure that the calculated start and end
    # indices match the reported span text. If this assert fails, it is likely
    # a bug in the data preparation code above.
    assert expected == answer["span_text"], (expected, answer["span_text"])

  return example


def make_nq_answer(contexts, answer):
  """Makes an Answer object following NQ conventions.

  Args:
    contexts: string containing the context
    answer: dictionary with `span_start` and `input_text` fields

  Returns:
    an Answer object. If the Answer type is YES or NO or LONG, the text
    of the answer is the long answer. If the answer type is UNKNOWN, the text of
    the answer is empty.
  """
  start = answer["span_start"]
  end = answer["span_end"]
  input_text = answer["input_text"]
  sa_entities = answer["sa_entities"]

  if (answer["candidate_id"] == -1 or start >= len(contexts) or
      end > len(contexts)):
    answer_type = AnswerType.UNKNOWN
    start = 0
    end = 1
  elif input_text.lower() == "yes":
    answer_type = AnswerType.YES
  elif input_text.lower() == "no":
    answer_type = AnswerType.NO
  elif input_text.lower() == "long":
    answer_type = AnswerType.LONG
  else:
    answer_type = AnswerType.SHORT

  return Answer(answer_type, text=contexts[start:end], offset=start, entities=sa_entities)


def read_nq_entry(entry, is_training):
  """Converts a NQ entry into a list of NqExamples."""

  def is_whitespace(c):
    return c in " \t\r\n" or ord(c) == 0x202F

  examples = []
  contexts_id = entry["id"]
  contexts = entry["contexts"]
  doc_tokens = []
  char_to_word_offset = []
  prev_is_whitespace = True
  for c in contexts:
    if is_whitespace(c):
      prev_is_whitespace = True
    else:
      if prev_is_whitespace:
        doc_tokens.append(c)
      else:
        doc_tokens[-1] += c
      prev_is_whitespace = False
    char_to_word_offset.append(len(doc_tokens) - 1)

  questions = []
  questions_emap = []
  for i, question in enumerate(entry["questions"]):
    qas_id = "{}".format(contexts_id)
    question_text = question["input_text"]
    question_entity_map = question["entity_map"]
    start_position = None
    end_position = None
    answer = None
    if is_training or FLAGS.mask_non_entity_in_text:
      answer_dict = entry["answers"][i]
      answer = make_nq_answer(contexts, answer_dict)

      # For now, only handle extractive, yes, and no.
      if answer is None or answer.offset is None:
        continue
      start_position = char_to_word_offset[answer.offset]
      end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

      # Only add answers where the text can be exactly recovered from the
      # document. If this CAN'T happen it's likely due to weird Unicode
      # stuff so we will just skip the example.
      #
      # Note that this means for training mode, every example is NOT
      # guaranteed to be preserved.
      actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
      cleaned_answer_text = " ".join(
          tokenization.whitespace_tokenize(answer.text))
      if actual_text.find(cleaned_answer_text) == -1:
        tf.logging.warning("Could not find answer: '%s' vs. '%s'", actual_text,
                           cleaned_answer_text)
        continue

    questions.append(question_text)
    questions_emap.append(question_entity_map)
    example = NqExample(
        example_id=int(contexts_id),
        qas_id=qas_id,
        questions=questions[:],
        questions_entity_map=questions_emap[:],
        doc_tokens=doc_tokens,
        doc_tokens_map=entry.get("contexts_map", None),
        entity_list=entry["context_entity_list"],
        ner_entity_list=entry["context_ner_entity_list"],
        answer=answer,
        start_position=start_position,
        end_position=end_position,
        annotation=entry["annotation"])
    examples.append(example)
  return examples


def convert_examples_to_features(examples, tokenizer, is_training, output_fn, pretrain_file=None):
  """Converts a list of NqExamples into InputFeatures."""
  num_spans_to_ids = collections.defaultdict(list)
  mode = 'train' if is_training else 'dev'
  apr_obj = ApproximatePageRank(mode=mode, task_id=FLAGS.task_id,
                                shard_id=FLAGS.shard_split_id)

  for example in examples:
    example_index = example.example_id
    features, stats = convert_single_example(example, tokenizer, apr_obj, is_training, pretrain_file)
    num_spans_to_ids[len(features)].append(example.qas_id)

    for feature in features:
      feature.example_index = example_index
      feature.unique_id = feature.example_index + feature.doc_span_index
      output_fn(feature)

  return num_spans_to_ids


def check_is_max_context(doc_spans, cur_span_index, position):
  """Check if this is the 'max context' doc span for the token."""
  # Because of the sliding window approach taken to scoring documents, a single
  # token can appear in multiple documents. E.g.
  #  Doc: the man went to the store and bought a gallon of milk
  #  Span A: the man went to the
  #  Span B: to the store and bought
  #  Span C: and bought a gallon of
  #  ...
  #
  # Now the word 'bought' will have two scores from spans B and C. We only
  # want to consider the score with "maximum context", which we define as
  # the *minimum* of its left and right context (the *sum* of left and
  # right context will always be the same, of course).
  #
  # In the example the maximum context for 'bought' would be span C since
  # it has 1 left context and 3 right context, while span B has 4 left context
  # and 0 right context.
  best_score = None
  best_span_index = None
  for (span_index, doc_span) in enumerate(doc_spans):
    end = doc_span.start + doc_span.length - 1
    if position < doc_span.start:
      continue
    if position > end:
      continue
    num_left_context = position - doc_span.start
    num_right_context = end - position
    score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
    if best_score is None or score > best_score:
      best_score = score
      best_span_index = span_index

  return cur_span_index == best_span_index


def get_related_facts(doc_span, token_to_textmap_index, entity_list, apr_obj,
                      tokenizer, question_entity_map, answer=None, ner_entity_list=None,
                      all_doc_tokens=None, fp=None, override_shortest_path=False, use_passage_seeds=True,
                      use_question_seeds=False):
  """For a given doc span, use seed entities, do APR, return related facts.

  Args:
   doc_span: A document span dictionary holding spart start,
                   end and len details
   token_to_textmap_index: A list mapping tokens to their positions
                                 in full context
   entity_list: A list mapping every token to their
                       WikiData entity if present
   apr_obj: An ApproximatePageRank object
   tokenizer: BERT tokenizer

  Returns:
   nl_fact_tokens: Tokenized NL form of facts
  """

  question_entities = set()
  for start_idx in question_entity_map.keys():
      for sub_span in question_entity_map[start_idx]:
          question_entities.add(sub_span[1])
  question_entities = list(question_entities)

  question_entity_ids = [int(apr_obj.data.ent2id[x]) for x in question_entities if x in apr_obj.data.ent2id]
  question_entity_names = str([apr_obj.data.entity_names['e'][str(x)]['name'] for x in question_entity_ids])

  answer_entity_ids = [int(apr_obj.data.ent2id[x]) for x in answer.entities if x in apr_obj.data.ent2id]
  answer_entity_names = str([apr_obj.data.entity_names['e'][str(x)]['name'] for x in answer_entity_ids])

  num_hops = None
  if FLAGS.use_shortest_path_facts and not override_shortest_path:
      facts, num_hops = apr_obj.get_shortest_path_facts(question_entities, answer.entities, passage_entities=[], seed_weighting=True, fp=fp)

      if len(facts)>0 and FLAGS.add_random_question_facts_to_shortest_path:
            facts.extend(apr_obj.get_random_facts_of_question(question_entities, answer.entities, passage_entities=[], seed_weighting=True, fp=fp))
      if len(facts)>0 and FLAGS.add_random_walk_question_facts_to_shortest_path:
          unique_facts = apr_obj.get_facts(question_entities, topk=200, alpha=FLAGS.alpha, seed_weighting=True)
          sorted_facts = sorted(unique_facts, key=lambda tup: tup[1][1], reverse=True)
          if FLAGS.num_facts_limit > 0:
              sorted_facts = sorted_facts[0:FLAGS.num_facts_limit]
          facts.extend(sorted_facts)

      if FLAGS.shuffle_shortest_path_facts:
          random.shuffle(facts)
      if FLAGS.use_entity_markers:
          nl_facts = " . ".join([
              "[unused0] " + str(x[0][0][1]) + " [unused1] " + str(x[1][0][1]) + " [unused0] " + str(x[0][1][1])
              for x in facts
          ])
      else:
          nl_facts = " . ".join([
              str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
              for x in facts
          ])
  else:
      # Adding this check since empty seeds generate random facts
      seed_entities = []
      if use_passage_seeds:
          start_index = token_to_textmap_index[doc_span.start]
          end_index = token_to_textmap_index[min(
              doc_span.start + doc_span.length - 1,
              len(token_to_textmap_index) -
              1)]  # putting this min check need to check all this later

          sub_list = entity_list[start_index:end_index + 1]
          sub_ner_list = ner_entity_list[start_index:end_index+1]
          sub_tokens = all_doc_tokens[start_index:end_index + 1]

          if FLAGS.use_named_entities_to_filter:
            seed_entities.extend([x[2:] for idx, x in enumerate(sub_list) if x.startswith('B-') and sub_ner_list[idx] != 'None'])
          else:
            seed_entities.extend([x[2:] for idx, x in enumerate(sub_list) if x.startswith('B-')])
      if use_question_seeds:
        if FLAGS.verbose_logging:
            print('Question seed entities: '+str(list(question_entities)))
        seed_entities.extend(list(question_entities))
      if not use_passage_seeds and not use_question_seeds:
        print("One of use_question or use passage seeds must be set. Wrong Usage!")
        exit()

      if seed_entities:
        if FLAGS.use_random_fact_generator:
            unique_facts = apr_obj.get_random_facts(
                seed_entities, topk=200, alpha=FLAGS.alpha, seed_weighting=True)
        else:
            unique_facts = apr_obj.get_facts(
                seed_entities, topk=200, alpha=FLAGS.alpha, seed_weighting=True)

        facts = sorted(unique_facts, key=lambda tup: tup[1][1], reverse=True)
        if FLAGS.num_facts_limit > 0:
            facts = facts[0:FLAGS.num_facts_limit]
        if FLAGS.use_entity_markers:
            nl_facts = " . ".join([
                "[unused0] " + str(x[0][0][1]) + " [unused1] " + str(x[1][0][1]) + " [unused0] " + str(x[0][1][1])
                for x in facts
            ])
        else:
            nl_facts = " . ".join([
                str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
                for x in facts
            ])
      else:
        nl_facts = ""

  #print(nl_facts[0:1000])
  # Tokening retrieved facts
  tok_to_orig_index = []
  tok_to_textmap_index = []
  orig_to_tok_index = []
  nl_fact_tokens = []
  for (i, token) in enumerate(nl_facts.split()):
    orig_to_tok_index.append(len(nl_fact_tokens))
    sub_tokens = tokenize(tokenizer, token)
    tok_to_textmap_index.extend([i] * len(sub_tokens))
    tok_to_orig_index.extend([i] * len(sub_tokens))
    nl_fact_tokens.extend(sub_tokens)
  return nl_fact_tokens, num_hops, question_entity_names, answer_entity_names

def get_all_question_answer_paths(apr_obj,
                      tokenizer, question_entity_map, answer=None, ner_entity_list=None,
                      all_doc_tokens=None, fp=None):
    """For a given doc span, use seed entities, do APR, return related facts.

    Args:
     doc_span: A document span dictionary holding spart start,
                     end and len details
     token_to_textmap_index: A list mapping tokens to their positions
                                   in full context
     entity_list: A list mapping every token to their
                         WikiData entity if present
     apr_obj: An ApproximatePageRank object
     tokenizer: BERT tokenizer

    Returns:
     nl_fact_tokens: Tokenized NL form of facts
    """

    question_entities = set()
    for start_idx in question_entity_map.keys():
        for sub_span in question_entity_map[start_idx]:
            question_entities.add(sub_span[1])

    question_entity_ids = [int(apr_obj.data.ent2id[x]) for x in list(question_entities) if x in apr_obj.data.ent2id]
    question_entity_names = str([apr_obj.data.entity_names['e'][str(x)]['name'] for x in question_entity_ids])

    answer_entity_ids = [int(apr_obj.data.ent2id[x]) for x in answer.entities if x in apr_obj.data.ent2id]
    answer_entity_names = str([apr_obj.data.entity_names['e'][str(x)]['name'] for x in answer_entity_ids])
    
    num_hops = None
    facts, num_hops = apr_obj.get_all_path_facts(list(question_entities), answer.entities, passage_entities=[], seed_weighting=True, fp=fp)

    nl_facts = [" . ".join([
            str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
            for x in single_path
        ]) for single_path in facts]
    #print(nl_facts)
    return nl_facts, num_hops, question_entity_names, answer_entity_names

def get_all_question_passage_paths(doc_span, token_to_textmap_index, entity_list, apr_obj,
                                  tokenizer, question_entity_map, answer=None, ner_entity_list=None,
                                  all_doc_tokens=None, fp=None):
    """For a given doc span, use seed entities, do APR, return related facts.

    Args:
     doc_span: A document span dictionary holding spart start,
                     end and len details
     token_to_textmap_index: A list mapping tokens to their positions
                                   in full context
     entity_list: A list mapping every token to their
                         WikiData entity if present
     apr_obj: An ApproximatePageRank object
     tokenizer: BERT tokenizer

    Returns:
     nl_fact_tokens: Tokenized NL form of facts
    """

    question_entities = set()
    for start_idx in question_entity_map.keys():
        for sub_span in question_entity_map[start_idx]:
            question_entities.add(sub_span[1])

    question_entity_ids = [int(apr_obj.data.ent2id[x]) for x in list(question_entities) if x in apr_obj.data.ent2id]
    question_entity_names = str([apr_obj.data.entity_names['e'][str(x)]['name'] for x in question_entity_ids])

    answer_entity_ids = [int(apr_obj.data.ent2id[x]) for x in answer.entities if x in apr_obj.data.ent2id]
    answer_entity_names = str([apr_obj.data.entity_names['e'][str(x)]['name'] for x in answer_entity_ids])

    start_index = token_to_textmap_index[doc_span.start]
    end_index = token_to_textmap_index[min( doc_span.start + doc_span.length - 1,
                                        len(token_to_textmap_index) - 1)]  # putting this min check need to check all this later
    sub_list = entity_list[start_index:end_index + 1]
    sub_ner_list = ner_entity_list[start_index:end_index+1]
    sub_tokens = all_doc_tokens[start_index:end_index + 1]
    if FLAGS.use_named_entities_to_filter:
        passage_entities = [x[2:] for idx, x in enumerate(sub_list) if x.startswith('B-') and sub_ner_list[idx] != 'None']
    else:
        passage_entities = [x[2:] for idx, x in enumerate(sub_list) if x.startswith('B-')]

    num_hops = None
    facts, num_hops = apr_obj.get_question_to_passage_facts(list(question_entities), answer.entities, passage_entities=passage_entities, seed_weighting=True, fp=fp)

    nl_facts = " . ".join([" . ".join([
        str(x[0][0][1]) + " " + str(x[1][0][1]) + " " + str(x[0][1][1])
        for x in single_path
    ]) for single_path in facts])

    tok_to_orig_index = []
    tok_to_textmap_index = []
    orig_to_tok_index = []
    nl_fact_tokens = []
    for (i, token) in enumerate(nl_facts.split()):
        orig_to_tok_index.append(len(nl_fact_tokens))
        sub_tokens = tokenize(tokenizer, token)
        tok_to_textmap_index.extend([i] * len(sub_tokens))
        tok_to_orig_index.extend([i] * len(sub_tokens))
        nl_fact_tokens.extend(sub_tokens)

    return nl_fact_tokens, num_hops, question_entity_names, answer_entity_names

def convert_single_example(example, tokenizer, apr_obj, is_training, pretrain_file=None, fixed_train_list=None):
    """Converts a single NqExample into a list of InputFeatures."""
    if FLAGS.use_question_level_apr_data:
        apr_obj = ApproximatePageRank(question_id=example.example_id)
    tok_to_orig_index = []
    tok_to_textmap_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    features = []
    extended_entity_list = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenize(tokenizer, token)
        tok_to_textmap_index.extend([i] * len(sub_tokens))
        extended_entity_list.extend([example.entity_list[i]] * len(sub_tokens))
        #print(sub_tokens, [example.entity_list[i]] * len(sub_tokens))
        tok_to_orig_index.extend([i] * len(sub_tokens))
        all_doc_tokens.extend(sub_tokens)
    # `tok_to_orig_index` maps wordpiece indices to indices of whitespace
    # tokenized word tokens in the contexts. The word tokens might themselves
    # correspond to word tokens in a larger document, with the mapping given
    # by `doc_tokens_map`.
    if example.doc_tokens_map:
        tok_to_orig_index = [
            example.doc_tokens_map[index] for index in tok_to_orig_index
        ]

    # QUERY
    query_tokens = []
    query_tokens.append("[Q]")
    query_tokens.extend(tokenize(tokenizer, example.questions[-1]))
    if len(query_tokens) > FLAGS.max_query_length:
        query_tokens = query_tokens[-FLAGS.max_query_length:]

    # ANSWER
    tok_start_position = 0
    tok_end_position = 0
    if is_training or FLAGS.mask_non_entity_in_text:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
    # if tok_start_position > tok_end_position : 
    #     print(example.start_position, example.end_position)
    #     print(tok_start_position, tok_end_position)
    #     exit()

    if FLAGS.create_fact_annotation_data and is_training:
        aligned_facts, num_hops, question_entity_names, answer_entity_names = get_all_question_answer_paths(apr_obj,
                                                               tokenizer, example.question_entity_map[-1], example.answer,
                                                               example.ner_entity_list, example.doc_tokens, pretrain_file)
        a = example.annotation
        la_text, sa_text = "", ""
        if a is not None:
            idx = a["long_answer"]["candidate_index"]
            la_text = a["long_answer"]["text_answer"]
            if a["short_answers"]:
                start_token = a["short_answers"][0]["start_token"]
                end_token = a["short_answers"][-1]["end_token"]
                sa_text = a["short_answers"][0]["text_answer"]
        aligned_facts = list(set(aligned_facts))
        for path in aligned_facts:
            if path != '':
                pretrain_file.write(" ".join(query_tokens).replace(" ##", "")+"\t"
                                    +str(question_entity_names)+"\t"
                                    +str(la_text)+"\t"
                                    +str(sa_text)+"\t"
                                    +str(answer_entity_names)+"\t"
                                    +str(path)+"\n")

    # The -4 accounts for [CLS], [SEP] and [SEP] and [SEP]
    max_tokens_for_doc = FLAGS.max_seq_length - len(query_tokens) - 4
    if FLAGS.augment_facts:
        max_tokens_for_para = int(max_tokens_for_doc / 2)
    else:
        max_tokens_for_para = int(max_tokens_for_doc)

    # max_tokens_for_facts = max_tokens_for_doc - max_tokens_for_para

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset  # remaining_len
        length = min(length, max_tokens_for_para)
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, FLAGS.doc_stride)

    valid_count = 0
    dev_valid_pos_answers = 0
    ent_dict = {}
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        # tf.logging.info("Processing Instance")

        contains_an_annotation = False
        if is_training or FLAGS.mask_non_entity_in_text:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            contains_an_annotation = (
                    tok_start_position >= doc_start and tok_end_position <= doc_end)
            span_id = (example.example_id + doc_span_index)
            if is_training and FLAGS.use_fixed_training_data:
                if span_id not in fixed_train_list:
                    continue
            elif ((not contains_an_annotation) or
                    example.answer.type == AnswerType.UNKNOWN):
                # If an example has unknown answer type or does not contain the answer
                # span, then we only include it with probability --include_unknowns.
                # When we include an example with unknown answer type, we set the first
                # token of the passage to be the annotated short span.
                # span_id = (example.example_id + doc_span_index)
                if is_training and (FLAGS.include_unknowns < 0 or
                        random.random() > FLAGS.include_unknowns):
                    print("How here")
                    continue
            else:
                pass
        # tf.logging.info("Processing Instance")
        tokens = []
        text_only_tokens = []
        masked_text_tokens = []
        masked_text_tokens_with_facts = []
        anonymized_text_only_tokens = []
 
        tmp_eval = []
        num_hops = None

        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        text_only_tokens.append("[CLS]")
        masked_text_tokens.append("[CLS]")
        masked_text_tokens_with_facts.append("[CLS]")
        anonymized_text_only_tokens.append("[CLS]")
        tmp_eval.append("[CLS]")

        segment_ids.append(0)
        tokens.extend(query_tokens)
        text_only_tokens.extend(query_tokens)
        masked_text_tokens.extend(query_tokens)
        masked_text_tokens_with_facts.extend(query_tokens)
        anonymized_text_only_tokens.extend(query_tokens)
        tmp_eval.extend(["None"]*len(query_tokens))
        segment_ids.extend([0] * len(query_tokens))

        tokens.append("[SEP]")
        text_only_tokens.append("[SEP]")
        masked_text_tokens.append("[SEP]")
        masked_text_tokens_with_facts.append("[SEP]")
        anonymized_text_only_tokens.append("[SEP]")
        tmp_eval.append("[SEP]")
        segment_ids.append(0)

        text_tokens = []
        fact_tokens = []
        answer_version = []
        ent_count = 0
        feature_stats = {}
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            textmap_idx = tok_to_textmap_index[split_token_index]
            e_val = example.entity_list[textmap_idx]
            if FLAGS.use_named_entities_to_filter:
                try:
                    e_ner_val = example.ner_entity_list[textmap_idx]
                except:
                    print(len(example.entity_list))
                    print(len(example.ner_entity_list))
                    print(textmap_idx)
                    exit()
                if e_ner_val == 'None':
                    e_val = 'None'

            is_max_context = check_is_max_context(doc_spans, doc_span_index,
                                                  split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            text_only_tokens.append(all_doc_tokens[split_token_index])
            text_tokens.append(all_doc_tokens[split_token_index])
            tmp_eval.append(e_val)
            if FLAGS.mask_non_entity_in_text and e_val == 'None':
                masked_text_tokens.append('[PAD]')
                masked_text_tokens_with_facts.append('[PAD]')
            else:
                masked_text_tokens.append(all_doc_tokens[split_token_index])
                masked_text_tokens_with_facts.append(all_doc_tokens[split_token_index])
            if FLAGS.anonymize_entities and e_val != 'None':
                if e_val.startswith('B-') and ent_count<98:
                    ent_count+=1
                anonymized_text_only_tokens.append('[unused'+str(ent_count)+']')
            else:
                anonymized_text_only_tokens.append(all_doc_tokens[split_token_index])
            #print(e_val, all_doc_tokens[split_token_index], tokens[-1])
            if FLAGS.mask_non_entity_in_text :
                if contains_an_annotation and (split_token_index>=tok_start_position and split_token_index<=tok_end_position):
                    answer_version.append(masked_text_tokens[-1])
            segment_ids.append(1)

        if ent_count in ent_dict:
            ent_dict[ent_count] += 1
        else:
            ent_dict[ent_count] = 1

        tokens.append("[SEP]")
        text_only_tokens.append("[SEP]")
        masked_text_tokens.append("[SEP]")
        masked_text_tokens_with_facts.append("[SEP]")
        anonymized_text_only_tokens.append("[SEP]")
        tmp_eval.append("[SEP]")
        segment_ids.append(1)
        
        if FLAGS.mask_non_entity_in_text and contains_an_annotation and (len(answer_version)==0 or '[PAD]' in answer_version):
            continue
        elif FLAGS.mask_non_entity_in_text and contains_an_annotation:
            dev_valid_pos_answers += 1
        else:
            pass
        # print(" ".join(tokens)+"\n"+" ".join(masked_text_tokens)+"\n"+" ".join(tmp_eval)+"\n\n")
        valid_count += 1
        #print(" ".join(text_tokens).replace(" ##", ""))
        if FLAGS.create_pretrain_data:
            pretrain_file.write(" ".join(text_tokens).replace(" ##", "")+"\n")
        # if FLAGS.create_fact_annotation_data:
        #     pretrain_file.write(" ".join(query_tokens).replace(" ##", "")+"\t"+" ".join(text_tokens).replace(" ##", "")+"\t")
        if FLAGS.augment_facts:
            #pretrain_file.write(example.questions[-1]+"\t"+" ".join(answer_version)+"\t")
            if FLAGS.verbose_logging:
                print(example.questions[-1])
                print(answer_version)
            aligned_facts_subtokens, num_hops, _, _ = get_related_facts(doc_span, tok_to_textmap_index,
                                                        example.entity_list, apr_obj,
                                                        tokenizer, example.question_entity_map[-1], example.answer,
                                                        example.ner_entity_list, example.doc_tokens, pretrain_file)
            if len(aligned_facts_subtokens) == 0 :
                if FLAGS.mask_non_entity_in_text and contains_an_annotation:
                    dev_valid_pos_answers -= 1
                continue
            if FLAGS.use_passage_rw_facts_in_shortest_path:
                aligned_facts_subtokens, _, _, _ = get_related_facts(doc_span, tok_to_textmap_index,
                                                                      example.entity_list, apr_obj,
                                                                      tokenizer, example.question_entity_map[-1], example.answer,
                                                                      example.ner_entity_list, example.doc_tokens, pretrain_file,
                                                                      override_shortest_path=True)
                if FLAGS.verbose_logging:
                    print("Newly aligned Facts")
                    print(aligned_facts_subtokens)
            if FLAGS.use_question_rw_facts_in_shortest_path:
                aligned_facts_subtokens, _, _, _ = get_related_facts(doc_span, tok_to_textmap_index,
                                                                      example.entity_list, apr_obj,
                                                                      tokenizer, example.question_entity_map[-1], example.answer,
                                                                      example.ner_entity_list, example.doc_tokens, pretrain_file,
                                                                      override_shortest_path=True, use_passage_seeds=False,
                                                                      use_question_seeds=True)
                if FLAGS.verbose_logging:
                    print("Newly aligned Facts")
                    print(aligned_facts_subtokens)
            if FLAGS.use_question_to_passage_facts_in_shortest_path:
                aligned_facts_subtokens, _, _, _ = get_all_question_passage_paths(doc_span, tok_to_textmap_index,
                                                                            example.entity_list, apr_obj,
                                                                            tokenizer, example.question_entity_map[-1],
                                                                            example.answer,
                                                                            example.ner_entity_list, example.doc_tokens,
                                                                            pretrain_file)
                if FLAGS.verbose_logging:
                    print("Newly aligned Facts")
                    print(aligned_facts_subtokens)
            max_tokens_for_current_facts = max_tokens_for_doc - doc_span.length
            for (index, token) in enumerate(aligned_facts_subtokens):
                if index >= max_tokens_for_current_facts:
                    break
                token_to_orig_map[len(tokens)] = -1  # check if this makes sense
                tokens.append(token)
                masked_text_tokens_with_facts.append(token)
                fact_tokens.append(token)
                segment_ids.append(1)
        tokens.append("[SEP]")
        text_only_tokens.append("[SEP]")
        masked_text_tokens.append("[SEP]")
        masked_text_tokens_with_facts.append("[SEP]")
        anonymized_text_only_tokens.append("[SEP]")
        segment_ids.append(1)
        assert len(tokens) == len(segment_ids)

        if FLAGS.create_pretrain_data:
            pretrain_file.write(" ".join(fact_tokens).replace(" ##", "")+"\n\n")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        text_only_input_ids = tokenizer.convert_tokens_to_ids(text_only_tokens)
        masked_text_tokens_input_ids = tokenizer.convert_tokens_to_ids(masked_text_tokens)
        masked_text_tokens_with_facts_input_ids = tokenizer.convert_tokens_to_ids(masked_text_tokens_with_facts)
        #print(tokens)
        #print(masked_text_tokens_input_ids)
        #print(anonymized_text_only_tokens)
        anonymized_text_only_tokens_input_ids = tokenizer.convert_tokens_to_ids(anonymized_text_only_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        #input_mask = [1] * len(input_ids) Since we now can have 0/PAD in between due to masking non-entity tokens

        input_mask = [0 if item == 0 else 1 for item in input_ids]

        # Zero-pad up to the sequence length.
        padding = [0] * (FLAGS.max_seq_length - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)

        text_only_tokens_mask = []
        masked_text_tokens_mask = None
        masked_text_tokens_with_facts_mask =None
        anonymized_text_only_tokens_mask = None

        if FLAGS.mask_non_entity_in_text:
            text_only_tokens_mask = [0 if item == 0 else 1 for item in text_only_input_ids]
            masked_text_tokens_mask = [0 if item == 0 else 1 for item in masked_text_tokens_input_ids]
            masked_text_tokens_with_facts_mask = [0 if item == 0 else 1 for item in masked_text_tokens_with_facts_input_ids]

            padding = [0] * (FLAGS.max_seq_length - len(text_only_input_ids))
            text_only_input_ids.extend(padding)
            text_only_tokens_mask.extend(padding)

            padding = [0] * (FLAGS.max_seq_length - len(masked_text_tokens_input_ids))
            masked_text_tokens_input_ids.extend(padding)
            masked_text_tokens_mask.extend(padding)

            padding = [0] * (FLAGS.max_seq_length - len(masked_text_tokens_with_facts_input_ids))
            masked_text_tokens_with_facts_input_ids.extend(padding)
            masked_text_tokens_with_facts_mask.extend(padding)

            assert len(masked_text_tokens_input_ids) == FLAGS.max_seq_length
            assert len(masked_text_tokens_mask) == FLAGS.max_seq_length
            assert len(masked_text_tokens_with_facts_input_ids) == FLAGS.max_seq_length
            assert len(masked_text_tokens_with_facts_mask) == FLAGS.max_seq_length

        if FLAGS.anonymize_entities:
            anonymized_text_only_tokens_mask = [0 if item == 0 else 1 for item in anonymized_text_only_tokens_input_ids]
            padding = [0] * (FLAGS.max_seq_length - len(anonymized_text_only_tokens_input_ids))
            anonymized_text_only_tokens_input_ids.extend(padding)
            anonymized_text_only_tokens_mask.extend(padding)

        # tf.logging.info('Len input_ids : %d', len(input_ids))
        # tf.logging.info('Max Seq Len : %d', FLAGS.max_seq_length)
        # tf.logging.info('Max tokens facts : %d', max_tokens_for_current_facts)
        # tf.logging.info('Max tokens para : %d', max_tokens_for_para)
        assert len(input_ids) == FLAGS.max_seq_length
        assert len(input_mask) == FLAGS.max_seq_length
        assert len(segment_ids) == FLAGS.max_seq_length

        start_position = None
        end_position = None
        answer_type = None
        answer_text = ""
        if is_training:
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            contains_an_annotation = (
                    tok_start_position >= doc_start and tok_end_position <= doc_end)
            if ((not contains_an_annotation) or
                    example.answer.type == AnswerType.UNKNOWN):
                # If an example has unknown answer type or does not contain the answer
                # span, then we only include it with probability --include_unknowns.
                # When we include an example with unknown answer type, we set the first
                # token of the passage to be the annotated short span.

                # if (FLAGS.include_unknowns < 0 or
                #     random.random() > FLAGS.include_unknowns):
                #   continue
                start_position = 0
                end_position = 0
                answer_type = AnswerType.UNKNOWN
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset
                answer_type = example.answer.type

            answer_text = " ".join(tokens[start_position:(end_position + 1)])
        if num_hops == 0:
            print("How 0 when adding?")
            exit()
        feature = InputFeatures(
            unique_id=-1,
            example_index=-1,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            text_only_input_ids=text_only_input_ids,
            text_only_tokens_mask=text_only_tokens_mask,
            masked_text_tokens_input_ids=masked_text_tokens_input_ids,
            masked_text_tokens_with_facts_input_ids=masked_text_tokens_with_facts_input_ids,
            masked_text_tokens_mask=masked_text_tokens_mask,
            masked_text_tokens_with_facts_mask=masked_text_tokens_with_facts_mask,
            anonymized_text_only_tokens_input_ids=anonymized_text_only_tokens_input_ids,
            anonymized_text_only_tokens_mask=anonymized_text_only_tokens_mask,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            answer_text=answer_text,
            answer_type=answer_type,
            num_hops=num_hops
        )  # Added facts to is max context and token to orig?
        features.append(feature)

    #print(ent_dict)
    #if sum([v for k, v in ent_dict.items()]) > 0:
    #    print(sum([k*v for k, v in ent_dict.items()])/sum([v for k, v in ent_dict.items()]))
    #else:
    #    print(0)

    if FLAGS.mask_non_entity_in_text and not is_training and dev_valid_pos_answers == 0:
        print('Dev example has no valid positive instances.')
        return [], None
    if FLAGS.create_fact_annotation_data and not is_training:
        aligned_facts, num_hops, question_entity_names, answer_entity_names = get_related_facts(None, tok_to_textmap_index,
                                                                       example.entity_list, apr_obj,
                                                                       tokenizer, example.question_entity_map[-1], example.answer,
                                                                       example.ner_entity_list, example.doc_tokens, pretrain_file)
        a = example.annotation
        la_text, sa_text = "", ""
        if a is not None:
            idx = a["long_answer"]["candidate_index"]
            la_text = a["long_answer"]["text_answer"]
            if a["short_answers"]:
                start_token = a["short_answers"][0]["start_token"]
                end_token = a["short_answers"][-1]["end_token"]
                sa_text = a["short_answers"][0]["text_answer"]
        # aligned_facts = list(set(aligned_facts))
        if aligned_facts != '':
            pretrain_file.write(str(example.example_id)+"\t"
                                    " ".join(query_tokens).replace(" ##", "")+"\t"
                                    +str(question_entity_names)+"\t"
                                    +str(la_text)+"\t"
                                    +str(sa_text)+"\t"
                                    +str(answer_entity_names)+"\t"
                                    +" ".join(aligned_facts).replace(" ##", "")+"\n")
                                    # +str(path)+"\n")
    return features, None


# A special token in NQ is made of non-space chars enclosed in square brackets.
_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)


def tokenize(tokenizer, text, apply_basic_tokenization=False):
  """Tokenizes text, optionally looking up special tokens separately.

  Args:
    tokenizer: a tokenizer from bert.tokenization.FullTokenizer
    text: text to tokenize
    apply_basic_tokenization: If True, apply the basic tokenization. If False,
      apply the full tokenization (basic + wordpiece).

  Returns:
    tokenized text.

  A special token is any text with no spaces enclosed in square brackets with no
  space, so we separate those out and look them up in the dictionary before
  doing actual tokenization.
  """
  tokenize_fn = tokenizer.tokenize
  if apply_basic_tokenization:
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
  tokens = []
  for token in text.split(" "):
    if _SPECIAL_TOKENS_RE.match(token):
      if token in tokenizer.vocab:
        tokens.append(token)
      else:
        tokens.append(tokenizer.wordpiece_tokenizer.unk_token)
    else:
      tokens.extend(tokenize_fn(token))
  return tokens


class CreateTFExampleFn(object):
  """Functor for creating NQ tf.Examples."""

  def __init__(self, is_training):
    self.is_training = is_training
    self.tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    mode = 'train' if is_training else 'dev'
    self.apr_obj = ApproximatePageRank(mode=mode, task_id=FLAGS.task_id,
                                       shard_id=FLAGS.shard_split_id)

  def process(self, example, pretrain_file=None, fixed_train_list=None):
    """Coverts an NQ example in a list of serialized tf examples."""
    nq_examples = read_nq_entry(example, self.is_training)
    input_features = []
    for nq_example in nq_examples:
      features, stat_counts = convert_single_example(nq_example, self.tokenizer, self.apr_obj,
                                                self.is_training, pretrain_file, fixed_train_list)
      input_features.extend(features)

    for input_feature in input_features:
      input_feature.example_index = int(example["id"])
      input_feature.unique_id = (
          input_feature.example_index + input_feature.doc_span_index)

      def create_int_feature(values):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))

      features = collections.OrderedDict()
      features["unique_ids"] = create_int_feature([input_feature.unique_id])
      features["input_ids"] = create_int_feature(input_feature.input_ids)
      features["input_mask"] = create_int_feature(input_feature.input_mask)
      features["segment_ids"] = create_int_feature(input_feature.segment_ids)

      if FLAGS.mask_non_entity_in_text:
        features["text_only_input_ids"] = create_int_feature(input_feature.text_only_input_ids)
        features["text_only_mask"] = create_int_feature(input_feature.text_only_mask)
        features["masked_text_tokens_input_ids"] = create_int_feature(input_feature.masked_text_tokens_input_ids)
        features["masked_text_tokens_mask"] = create_int_feature(input_feature.masked_text_tokens_mask)
        features["masked_text_tokens_with_facts_input_ids"] = create_int_feature(input_feature.masked_text_tokens_with_facts_input_ids)
        features["masked_text_tokens_with_facts_mask"] = create_int_feature(input_feature.masked_text_tokens_with_facts_mask)

      if FLAGS.anonymize_entities:
        features["anonymized_text_only_tokens_input_ids"] = create_int_feature(input_feature.anonymized_text_only_tokens_input_ids)
        features["anonymized_text_only_tokens_mask"] = create_int_feature(input_feature.anonymized_text_only_tokens_mask)

      if FLAGS.use_shortest_path_facts:
        features["shortest_path_num_hops"] = create_int_feature([input_feature.num_hops])


      if self.is_training:
        features["start_positions"] = create_int_feature(
            [input_feature.start_position])
        features["end_positions"] = create_int_feature(
            [input_feature.end_position])
        features["answer_types"] = create_int_feature(
            [input_feature.answer_type])
      else:
        token_map = [-1] * len(input_feature.input_ids)
        for k, v in input_feature.token_to_orig_map.items():
          token_map[k] = v
        features["token_map"] = create_int_feature(token_map)

      yield tf.train.Example(features=tf.train.Features(
          feature=features)).SerializeToString()


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               tokens,
               token_to_orig_map,
               token_is_max_context,
               input_ids,
               input_mask,
               segment_ids,
               text_only_input_ids=None,
               text_only_tokens_mask=None,
               masked_text_tokens_input_ids=None,
               masked_text_tokens_with_facts_input_ids=None,
               masked_text_tokens_mask=None,
               masked_text_tokens_with_facts_mask=None,
               anonymized_text_only_tokens_input_ids=None,
               anonymized_text_only_tokens_mask=None,
               start_position=None,
               end_position=None,
               answer_text="",
               answer_type=AnswerType.SHORT,
               num_hops=None):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.tokens = tokens
    self.token_to_orig_map = token_to_orig_map
    self.token_is_max_context = token_is_max_context
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.text_only_input_ids=text_only_input_ids
    self.text_only_mask=text_only_tokens_mask
    self.masked_text_tokens_input_ids=masked_text_tokens_input_ids
    self.masked_text_tokens_with_facts_input_ids=masked_text_tokens_with_facts_input_ids
    self.masked_text_tokens_mask=masked_text_tokens_mask
    self.masked_text_tokens_with_facts_mask=masked_text_tokens_with_facts_mask
    self.anonymized_text_only_tokens_input_ids=anonymized_text_only_tokens_input_ids
    self.anonymized_text_only_tokens_mask=anonymized_text_only_tokens_mask
    self.start_position = start_position
    self.end_position = end_position
    self.answer_text = answer_text
    self.answer_type = answer_type
    self.num_hops = num_hops


def read_nq_examples(input_file, is_training):
  """Read a NQ json file into a list of NqExample."""
  input_paths = tf.gfile.Glob(input_file)
  input_data = []

  def _open(path):
    if path.endswith(".gz"):
      return gzip.GzipFile(fileobj=tf.gfile.Open(path, "rb"))
    else:
      return tf.gfile.Open(path, "r")

  for path in input_paths:
    tf.logging.info("Reading: %s", path)
    with _open(path) as input_file:
      for line in input_file:
        input_data.append(create_example_from_jsonl(line))

  examples = []
  for entry in input_data:
    examples.extend(read_nq_entry(entry, is_training))
  return examples


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # Get the logits for the start and end predictions.
  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
      "cls/nq/output_weights", [2, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "cls/nq/output_bias", [2], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [batch_size, seq_length, 2])
  logits = tf.transpose(logits, [2, 0, 1])

  unstacked_logits = tf.unstack(logits, axis=0)

  (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

  # Get the logits for the answer type prediction.
  answer_type_output_layer = model.get_pooled_output()
  answer_type_hidden_size = answer_type_output_layer.shape[-1].value

  num_answer_types = 5  # YES, NO, UNKNOWN, SHORT, LONG
  answer_type_output_weights = tf.get_variable(
      "answer_type_output_weights", [num_answer_types, answer_type_hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  answer_type_output_bias = tf.get_variable(
      "answer_type_output_bias", [num_answer_types],
      initializer=tf.zeros_initializer())

  answer_type_logits = tf.matmul(
      answer_type_output_layer, answer_type_output_weights, transpose_b=True)
  answer_type_logits = tf.nn.bias_add(answer_type_logits,
                                      answer_type_output_bias)

  return (start_logits, end_logits, answer_type_logits)


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    if FLAGS.mask_non_entity_in_text and FLAGS.use_text_only:
        tf.logging.info("Using text only full context tokens")
        input_ids = features["text_only_input_ids"]
        input_mask = features["text_only_mask"]
    elif FLAGS.mask_non_entity_in_text and FLAGS.use_masked_text_only:
        tf.logging.info("Using masked text only")
        input_ids = features["masked_text_tokens_input_ids"]
        input_mask = features["masked_text_tokens_mask"]
    elif FLAGS.mask_non_entity_in_text and FLAGS.use_text_and_facts:
        tf.logging.info("Using masked text with facts")
        input_ids = features["masked_text_tokens_with_facts_input_ids"]
        input_mask = features["masked_text_tokens_with_facts_mask"]
    elif FLAGS.mask_non_entity_in_text:
        print("use_text_only or use_text_and_facts must be set if masked versions")
        exit()
    elif FLAGS.anonymize_entities:
        tf.logging.info("Using anonymized entity version")
        input_ids = features["anonymized_text_only_tokens_input_ids"]
        input_mask = features["anonymized_text_only_tokens_mask"]
    else:
        pass
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits, answer_type_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]

      # Computes the loss for positions.
      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      # Computes the loss for labels.
      def compute_label_loss(logits, labels):
        one_hot_labels = tf.one_hot(
            labels, depth=len(AnswerType), dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]
      answer_types = features["answer_types"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)
      answer_type_loss = compute_label_loss(answer_type_logits, answer_types)

      total_loss = (start_loss + end_loss + answer_type_loss) / 3.0

      train_op = optimization.create_optimizer(total_loss, learning_rate,
                                               num_train_steps,
                                               num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      seq_length = modeling.get_shape_list(input_ids)[1]

      # Computes the loss for positions.
      def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

      # Computes the loss for labels.
      def compute_label_loss(logits, labels):
        one_hot_labels = tf.one_hot(
            labels, depth=len(AnswerType), dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_labels * log_probs, axis=-1))
        return loss

      start_positions = features["start_positions"]
      end_positions = features["end_positions"]
      answer_types = features["answer_types"]

      start_loss = compute_loss(start_logits, start_positions)
      end_loss = compute_loss(end_logits, end_positions)
      answer_type_loss = compute_label_loss(answer_type_logits, answer_types)

      total_loss = (start_loss + end_loss + answer_type_loss) / 3.0
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.PREDICT:  
      predictions = {
          "unique_ids": unique_ids,
          "start_logits": start_logits,
          "end_logits": end_logits,
          "answer_type_logits": answer_type_logits,
          "input_ids": input_ids,
          #"loss": total_loss,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and PREDICT modes are supported: %s" %
                       (mode))

    return output_spec

  return model_fn


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if FLAGS.mask_non_entity_in_text:
      name_to_features["text_only_input_ids"] = tf.FixedLenFeature([seq_length], tf.int64)
      name_to_features["text_only_mask"] = tf.FixedLenFeature([seq_length], tf.int64)
      name_to_features["masked_text_tokens_input_ids"] = tf.FixedLenFeature([seq_length], tf.int64)
      name_to_features["masked_text_tokens_with_facts_input_ids"] = tf.FixedLenFeature([seq_length], tf.int64)
      name_to_features["masked_text_tokens_mask"] = tf.FixedLenFeature([seq_length], tf.int64)
      name_to_features["masked_text_tokens_with_facts_mask"] = tf.FixedLenFeature([seq_length], tf.int64)

  if FLAGS.anonymize_entities:
      name_to_features["anonymized_text_only_tokens_input_ids"] = tf.FixedLenFeature([seq_length], tf.int64)
      name_to_features["anonymized_text_only_tokens_mask"] = tf.FixedLenFeature([seq_length], tf.int64)

  if is_training:
      name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
      name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)
      name_to_features["answer_types"] = tf.FixedLenFeature([], tf.int64)

  if FLAGS.use_shortest_path_facts:
      name_to_features["shortest_path_num_hops"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    # d = tf.data.TFRecordDataset(input_file)
    d = tf.data.Dataset.list_files(input_file, shuffle=False)
    d = d.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset, cycle_length=50, sloppy=is_training))
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


RawResult = collections.namedtuple(
    "RawResult",
    ["unique_id", "start_logits", "end_logits", "answer_type_logits"])


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    if filename is not None:
      self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1
    tf_example = self.get_processed_feature(feature)
    self._writer.write(tf_example.SerializeToString())

  def get_processed_feature(self, feature):
    """Return feature as an InputFeature."""
    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["unique_ids"] = create_int_feature([feature.unique_id])
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if FLAGS.mask_non_entity_in_text:
        features["text_only_input_ids"] = create_int_feature(feature.text_only_input_ids)
        features["text_only_mask"] = create_int_feature(feature.text_only_mask)
        features["masked_text_tokens_input_ids"] = create_int_feature(feature.masked_text_tokens_input_ids)
        features["masked_text_tokens_mask"] = create_int_feature(feature.masked_text_tokens_mask)
        features["masked_text_tokens_with_facts_input_ids"] = create_int_feature(feature.masked_text_tokens_with_facts_input_ids)
        features["masked_text_tokens_with_facts_mask"] = create_int_feature(feature.masked_text_tokens_with_facts_mask)
    if FLAGS.anonymize_entities:
        features["anonymized_text_only_tokens_input_ids"] = create_int_feature(feature.anonymized_text_only_tokens_input_ids)
        features["anonymized_text_only_tokens_mask"] = create_int_feature(feature.anonymized_text_only_tokens_mask)

    if FLAGS.use_shortest_path_facts:
        features["shortest_path_num_hops"] = create_int_feature([feature.num_hops])

    if self.is_training:
      features["start_positions"] = create_int_feature([feature.start_position])
      features["end_positions"] = create_int_feature([feature.end_position])
      features["answer_types"] = create_int_feature([feature.answer_type])
    else:
      token_map = [-1] * len(feature.input_ids)
      for k, v in feature.token_to_orig_map.items():
        token_map[k] = v
      features["token_map"] = create_int_feature(token_map)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    return tf_example

  def close(self):
    self._writer.close()


Span = collections.namedtuple("Span", ["start_token_idx", "end_token_idx"])


class EvalExample(object):
  """Eval data available for a single example."""

  def __init__(self, example_id, candidates):
    self.example_id = example_id
    self.candidates = candidates
    self.results = {}
    self.features = {}


class ScoreSummary(object):

  def __init__(self):
    self.predicted_label = None
    self.short_span_score = None
    self.cls_token_score = None
    self.answer_type_logits = None


def read_candidates_from_one_split(input_path):
  """Read candidates from a single jsonl file."""
  candidates_dict = {}
  with gzip.GzipFile(fileobj=tf.gfile.Open(input_path, "rb")) as input_file:
    tf.logging.info("Reading examples from: %s", input_path)
    for line in input_file:
      e = json.loads(line.decode('utf-8'))
      candidates_dict[e["example_id"]] = e["long_answer_candidates"]
  return candidates_dict


def read_candidates(input_pattern):
  """Read candidates with real multiple processes."""
  input_paths = tf.gfile.Glob(input_pattern)
  final_dict = {}
  for input_path in input_paths:
    final_dict.update(read_candidates_from_one_split(input_path))
  return final_dict


def get_best_indexes(logits, n_best_size):
  """Get the n-best logits from a list."""
  index_and_score = sorted(
      enumerate(logits[1:], 1), key=lambda x: x[1], reverse=True)
  best_indexes = []
  for i in range(len(index_and_score)):
    if i >= n_best_size:
      break
    best_indexes.append(index_and_score[i][0])
  return best_indexes


def compute_predictions(example, tokenizer = None, pred_fp = None):
  """Converts an example into an NQEval object for evaluation."""
  predictions = []
  n_best_size = 10
  max_answer_length = 30
  num_hops = None

  for unique_id, result in example.results.items():
    if unique_id not in example.features:
      raise ValueError("No feature found with unique_id:", unique_id)
    token_map = example.features[unique_id]["token_map"].int64_list.value
    input_ids = example.features[unique_id]["input_ids"].int64_list.value
    masked_input_ids = []
    if FLAGS.mask_non_entity_in_text and FLAGS.use_text_only:
        masked_input_ids = example.features[unique_id]["text_only_input_ids"].int64_list.value
    if FLAGS.mask_non_entity_in_text and FLAGS.use_masked_text_only:
        masked_input_ids = example.features[unique_id]["masked_text_tokens_input_ids"].int64_list.value
    if FLAGS.mask_non_entity_in_text and FLAGS.use_text_and_facts:
        masked_input_ids = example.features[unique_id]["masked_text_tokens_with_facts_input_ids"].int64_list.value
    if FLAGS.anonymize_entities:
        masked_input_ids = example.features[unique_id]["anonymized_text_only_tokens_input_ids"].int64_list.value
    if FLAGS.use_shortest_path_facts:
        num_hops = example.features[unique_id]["shortest_path_num_hops"].int64_list.value[0]
    start_indexes = get_best_indexes(result["start_logits"], n_best_size)
    end_indexes = get_best_indexes(result["end_logits"], n_best_size)
    summary = None
    for start_index in start_indexes:
      for end_index in end_indexes:
        if end_index < start_index:
          continue
        if token_map[start_index] == -1:
          continue
        if token_map[end_index] == -1:
          continue
        length = end_index - start_index + 1
        if length > max_answer_length:
          continue
        summary = ScoreSummary()
        summary.short_span_score = (
            result["start_logits"][start_index] +
            result["end_logits"][end_index])
        summary.cls_token_score = (
            result["start_logits"][0] + result["end_logits"][0])
        summary.answer_type_logits = result["answer_type_logits"]
        start_span = token_map[start_index]
        end_span = token_map[end_index] + 1

        # Span logits minus the cls logits seems to be close to the best.
        score = summary.short_span_score - summary.cls_token_score
        predictions.append((score, summary, start_span, end_span, input_ids, masked_input_ids))
    if summary is None:
      # Hacking a summary where the first token of the context is the pred span
      summary = ScoreSummary()
      start_index = 0
      # Finding the first token that is actually from the context
      for idx, tokenmapval in enumerate(token_map):
        if tokenmapval != -1:
          start_index = idx
          break
      end_index = start_index
      summary.short_span_score = (
          result["start_logits"][start_index] + result["end_logits"][end_index])
      summary.cls_token_score = (
          result["start_logits"][0] + result["end_logits"][0])
      summary.answer_type_logits = result["answer_type_logits"]
      start_span = token_map[start_index]
      end_span = token_map[end_index] + 1
      # Span logits minus the cls logits seems to be close to the best.
      score = summary.short_span_score - summary.cls_token_score
      predictions.append((score, summary, start_span, end_span, input_ids, masked_input_ids))

  # tf.logging.info("Predictions list len : %d", len(predictions))
  # tf.logging.info("Len of example results: %d", len(example.results.items()))
  if len(predictions) == 0:
    summary = ScoreSummary()
    summary.short_span_score = (0)
    summary.cls_token_score = (0)
    #summary.answer_type_logits = result["answer_type_logits"]
    #start_span = token_map[start_index]
    #end_span = token_map[end_index] + 1
    #short_span = Span(-1, -1)
    #long_span = Span(-1, -1)
    start_span = -1
    end_span = -1
    score = 0
    predictions.append((score, summary, start_span, end_span, [], []))
  #else:
  #print(len(example.results.items()))
  score, summary, start_span, end_span, input_ids, masked_input_ids = sorted(
      predictions, reverse=True, key=lambda tup: tup[0])[0]
  short_span = Span(start_span, end_span)
  long_span = Span(-1, -1)
  for c in example.candidates:
    start = short_span.start_token_idx
    end = short_span.end_token_idx
    if c["top_level"] and c["start_token"] <= start and c["end_token"] >= end:
      long_span = Span(c["start_token"], c["end_token"])
      break

  input_ids = list(map(int, input_ids))
  if len(input_ids) > 0:
      input_text = tokenizer.convert_ids_to_tokens(input_ids)
      input_text = (" ".join(input_text)).replace(" ##","")
  else:
      input_text = ""

  masked_input_ids = list(map(int, masked_input_ids))
  if len(masked_input_ids) > 0:
      masked_input_text = tokenizer.convert_ids_to_tokens(masked_input_ids)
      masked_input_text = (" ".join(masked_input_text)).replace(" ##","")
  else:
      masked_input_text = ""

  summary.predicted_label = {
      "example_id": example.example_id,
      "long_answer": {
          "start_token": long_span.start_token_idx,
          "end_token": long_span.end_token_idx,
          "start_byte": -1,
          "end_byte": -1
      },
      "long_answer_score": score,
      "short_answers": [{
          "start_token": short_span.start_token_idx,
          "end_token": short_span.end_token_idx,
          "start_byte": -1,
          "end_byte": -1
      }],
      "short_answers_score": score,
      "yes_no_answer": "NONE",
      "input_ids": input_ids,
      "input_text": input_text,
      "masked_input_ids": masked_input_ids,
      "masked_input_text": masked_input_text,
      "shortest_path_num_hops": num_hops,
  }

  # if FLAGS.write_pred_analysis:
  #     input_ids = map(int, input_ids)
  #     question = []
  #     text = []
  #     facts = []
  #     current='question'
  #     for token in input_ids:
  #         try:
  #             word = tokenizer.convert_ids_to_tokens([token])[0]
  #             if current == 'question':
  #                 question.append(word)
  #             elif current == 'text':
  #                 text.append(word)
  #             elif current == 'facts':
  #                 facts.append(word)
  #             else:
  #                 print("Some exception in current word")
  #                 print(current)
  #             if word == '[SEP]' and current == 'question':
  #                 current = 'text'
  #             elif word == '[SEP]' and current == 'text':
  #                 current = 'facts'
  #             else:
  #                 continue
  #         except:
  #             print('didnt tokenize')
  #
  #     pred_fp.write(" ".join(question).replace(" ##","")+"\t")
  #     pred_fp.write(" ".join(text).replace(" ##", "")+"\t")
  #     pred_fp.write(" ".join(facts).replace(" ##","")+"\n")
  return summary


def compute_pred_dict(candidates_dict, dev_features, raw_results, tokenizer=None, pred_fp=None):
  """Computes official answer key from raw logits."""
  sess = tf.Session()
  # raw_results_by_id = [
  # (int(res["unique_id"] + 1), res) for res in raw_results]
  raw_results = [res for res in raw_results]
  raw_results_ids = tf.to_int32(
      np.array([int(res["unique_id"] + 1) for res in raw_results
               ])).eval(session=sess)
  raw_results_by_id = list(zip(raw_results_ids, raw_results))

  # Cast example id to int32 for each example, similarly to the raw results.
  all_candidates = candidates_dict.items()
  example_ids = tf.to_int32(np.array([int(k) for k, _ in all_candidates
                                     ])).eval(session=sess)
  examples_by_id = list(zip(example_ids, all_candidates))

  # Cast unique_id also to int32 for features.
  feature_ids = []
  features = []
  for f in dev_features:
    feature_ids.append(f.features.feature["unique_ids"].int64_list.value[0] + 1)
    features.append(f.features.feature)
  feature_ids = tf.to_int32(np.array(feature_ids)).eval(session=sess)
  features_by_id = list(zip(feature_ids, features))

  # Join examplew with features and raw results.
  examples = []
  merged = sorted(
      examples_by_id + raw_results_by_id + features_by_id,
      key=lambda tup: tup[0])
  for idx, datum in merged:
    if isinstance(datum, tuple):
      examples.append(EvalExample(datum[0], datum[1]))
    elif "token_map" in datum:
      examples[-1].features[idx] = datum
    else:
      examples[-1].results[idx] = datum

  # Construct prediction objects.
  print("Computing final summaries...")
  summary_dict = {}
  nq_pred_dict = {}
  for e in examples:
    if True or FLAGS.mask_non_entity_in_text:
        if len(list(e.features.keys())) == 0 and len(list(e.results)) == 0:
            continue
    summary = compute_predictions(e, tokenizer)
    summary_dict[e.example_id] = summary
    nq_pred_dict[e.example_id] = summary.predicted_label
    if len(nq_pred_dict) % 100 == 0:
      print("Examples processed: %d", len(nq_pred_dict))
  print("Done computing predictions.")

  return nq_pred_dict


def validate_flags_or_throw(bert_config):
  """Validate the input FLAGS or throw an exception."""
  if not FLAGS.do_train and not FLAGS.do_predict:
    raise ValueError("At least one of `{do_train,do_predict}` must be True.")

  if FLAGS.do_train:
    if not FLAGS.train_precomputed_file:
      raise ValueError("If `do_train` is True, then `train_precomputed_file` "
                       "must be specified.")
    if not FLAGS.train_num_precomputed:
      raise ValueError("If `do_train` is True, then `train_num_precomputed` "
                       "must be specified.")

  if FLAGS.do_predict:
    if not FLAGS.predict_file:
      raise ValueError(
          "If `do_predict` is True, then `predict_file` must be specified.")

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  if FLAGS.max_seq_length <= FLAGS.max_query_length + 3:
    raise ValueError(
        "The max_seq_length (%d) must be greater than max_query_length "
        "(%d) + 3" % (FLAGS.max_seq_length, FLAGS.max_query_length))


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  validate_flags_or_throw(bert_config)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  # Maintaining tokenizer for future use # pylint: disable=unused-variable
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    num_train_features = FLAGS.train_num_precomputed
    num_train_steps = int(num_train_features / FLAGS.train_batch_size *
                          FLAGS.num_train_epochs)

    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this falls back to normal Estimator on CPU or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.predict_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training on precomputed features *****")
    tf.logging.info("  Num split examples = %d", num_train_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
  train_filename = FLAGS.train_precomputed_file
  train_input_fn = input_fn_builder(
        input_file=train_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
  if FLAGS.do_train:  
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    if not FLAGS.output_prediction_file:
      raise ValueError(
          "--output_prediction_file must be defined in predict mode.")
    print("Evaluating 1000 steps now")
    estimator.evaluate(input_fn=train_input_fn, steps=1000)

    eval_filename = os.path.join(FLAGS.eval_data_path, "eval.tf-record")
    #eval_filename = FLAGS.eval_data_path

    tf.logging.info("***** Running predictions *****")

    predict_input_fn = input_fn_builder(
        input_file=eval_filename,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of steps.
    all_results = []
    loss = []
    for result in estimator.predict(
        predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      unique_id = int(result["unique_ids"])
      start_logits = [float(x) for x in result["start_logits"].flat]
      end_logits = [float(x) for x in result["end_logits"].flat]
      answer_type_logits = [float(x) for x in result["answer_type_logits"].flat]
      #loss.append(float(result['loss']))
      all_results.append(
          RawResult(
              unique_id=unique_id,
              start_logits=start_logits,
              end_logits=end_logits,
              answer_type_logits=answer_type_logits))

    #avg_loss = float(sum(loss))/len(loss)
    #print('Avg Loss: '+str(float(sum(loss))/len(loss)))
    candidates_dict = read_candidates(FLAGS.predict_file)
    eval_features = [
        tf.train.Example.FromString(r)
        for r in tf.python_io.tf_record_iterator(eval_filename)
    ]
    # eval_features = []
    print("Computing predictions")
    nq_pred_dict = compute_pred_dict(candidates_dict, eval_features,
                                     [r._asdict() for r in all_results], tokenizer, pred_fp=None)
    predictions_json = {"predictions": list(nq_pred_dict.values())}
    with tf.gfile.Open(FLAGS.output_prediction_file, "w") as f:
      json.dump(predictions_json, f, indent=4)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
