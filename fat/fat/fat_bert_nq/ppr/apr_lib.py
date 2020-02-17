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

"""This file contains a class which acts as a wrapper around the PPR algorithm.

This class has the following functionality:
1. Load the KB graph,
2. Given list of seed entities, get topk entities from PPR.
3. Get unique facts between all extracted entities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import tensorflow as tf
from fat.fat_bert_nq.ppr.apr_algo import csr_personalized_pagerank
from fat.fat_bert_nq.ppr.apr_algo import csr_topk_fact_extractor
from fat.fat_bert_nq.ppr.apr_algo import csr_get_k_hop_entities, csr_get_k_hop_facts
from fat.fat_bert_nq.ppr.kb_csr_io import CsrData

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_bool(
    'verbose_logging', False,
    'If true, all of the warnings related to data processing will be printed. '
    'A number of warnings are expected for a normal NQ evaluation.')


class ApproximatePageRank(object):
  """APR main lib which is used to wrap functions around ppr algo."""

  def __init__(self, mode=None, task_id=None, shard_id=None):
    self.data = CsrData()
    self.data.load_csr_data(
        full_wiki=FLAGS.full_wiki, files_dir=FLAGS.apr_files_dir,
        mode=mode, task_id=task_id, shard_id=shard_id)
    self.high_freq_relations = {'P31': 'instance of',
                                'P17': 'country',
                                'P131': 'located in the administrative territorial entity',
                                'P106': 'occupation',
                                'P21': 'sex or gender',
                                'P735': 'given name',
                                'P27': 'country of citizenship',
                                'P19': 'place of birth'}

  def get_khop_entities(self, seeds, k_hop):
    print("id2ent size: %d", len(self.data.id2ent))
    entity_ids = [
      int(self.data.ent2id[x]) for x in seeds if x in self.data.ent2id
    ]
    khop_entity_ids = csr_get_k_hop_entities(entity_ids, self.data.adj_mat_t_csr, k_hop)
    khop_entities = [
        self.data.id2ent[str(x)] for x in khop_entity_ids if str(x) in self.data.id2ent.keys()
    ]
    return khop_entities

  def get_khop_facts(self, seeds, k_hop):
      seeds = list(set(seeds))
      print("id2ent size: %d", len(self.data.id2ent))
      entity_ids = [
          int(self.data.ent2id[x]) for x in seeds if x in self.data.ent2id
      ]
      khop_entity_ids, khop_facts = csr_get_k_hop_facts(entity_ids, self.data.adj_mat_t_csr, self.data.rel_dict, k_hop)
      khop_entities = [
          self.data.id2ent[str(x)] for x in khop_entity_ids if str(x) in self.data.id2ent.keys()
      ]
      khop_facts = [
          ((self.data.id2ent[str(s)], self.data.entity_names['e'][str(s)]),
           (self.data.id2rel[r], self.data.entity_names['r'][str(r)]),
           (self.data.id2ent[str(o)], self.data.entity_names['e'][str(0)])) for (s,r,o) in khop_facts
      ]
      return khop_entities, khop_facts

  def get_topk_extracted_ent(self, seeds, alpha, topk):
    """Extract topk entities given seeds.

    Args:
      seeds: An Ex1 vector with weight on every seed entity
      alpha: probability for PPR
      topk: max top entities to extract
    Returns:
      extracted_ents: list of selected entities
      extracted_scores: list of scores of selected entities
    """
    #tf.logging.info('Start ppr')
    ppr_scores = csr_personalized_pagerank(seeds, self.data.adj_mat_t_csr,
                                           alpha)
    #tf.logging.info('End ppr')
    sorted_idx = np.argsort(ppr_scores)[::-1]
    extracted_ents = sorted_idx[:topk]
    extracted_scores = ppr_scores[sorted_idx[:topk]]

    # Check for really low values
    # Get idx of First value < 1e-6, limit extracted ents till there
    zero_idx = np.where(ppr_scores[extracted_ents] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
      extracted_ents = extracted_ents[:zero_idx[0]]

    return extracted_ents, extracted_scores

  def get_facts(self, entities, topk, alpha, seed_weighting=True):
    """Get subgraph describing a neighbourhood around given entities.

    Args:
      entities: A list of Wikidata entities
      topk: Max entities to extract from PPR
      alpha: Node probability for PPR
      seed_weighting: Boolean for performing weighting seeds by freq in passage

    Returns:
      unique_facts: A list of unique facts around the seeds.
    """

    if FLAGS.verbose_logging:
      print('Getting subgraph')
      tf.logging.info('Getting subgraph')
    entity_ids = [
        int(self.data.ent2id[x]) for x in entities if x in self.data.ent2id
    ]
    if FLAGS.verbose_logging:
      print(str([self.data.entity_names['e'][str(x)]['name'] for x in entity_ids
                                          ]))
      tf.logging.info(
          str([self.data.entity_names['e'][str(x)]['name'] for x in entity_ids
              ]))
    freq_dict = {x: entity_ids.count(x) for x in entity_ids}

    seed = np.zeros((self.data.adj_mat_t_csr.shape[0], 1))
    if not seed_weighting:
      seed[entity_ids] = 1. / len(set(entity_ids))
    else:
      for x, y in freq_dict.items():
        seed[x] = y
      seed = seed / seed.sum()

    extracted_ents, extracted_scores = self.get_topk_extracted_ent(
        seed, alpha, topk)
    if FLAGS.verbose_logging:
      print('Extracted Ents')
      tf.logging.info('Extracted ents: ')
      tf.logging.info(
          str([
              self.data.entity_names['e'][str(x)]['name']
              for x in extracted_ents
          ]))
      print(str([
                        self.data.entity_names['e'][str(x)]['name']
                                      for x in extracted_ents
                                                ][0:100]))

    facts = csr_topk_fact_extractor(self.data.adj_mat_t_csr, self.data.rel_dict,
                                    freq_dict, self.data.entity_names,
                                    extracted_ents, extracted_scores)
    if FLAGS.verbose_logging:
      #print('Extracted facts: ')
      #print(str(facts))
      tf.logging.info('Extracted facts: ')
      tf.logging.info(str(facts))

    # Extract 1 unique fact per pair of entities (fact with highest score)
    # Sort by scores
    unique_facts = {}
    for (sub, obj, rel, score) in facts:
      fwd_dir = (sub, obj, rel)
      rev_dir = (obj, sub, rel)
      if sub[1] == obj[1]: #No self-links
        continue
      #fwd_dir = (sub[1], obj[1])
      #rev_dir = (obj[1], sub[1])
      
      if fwd_dir in unique_facts:
        if score > unique_facts[fwd_dir][1]:
          unique_facts[fwd_dir] = (rel, score)
        else:
          continue
      elif rev_dir in unique_facts:
        if score > unique_facts[rev_dir][1]:
          unique_facts[fwd_dir] = (rel, score)
          del unique_facts[rev_dir]  # Remove existing entity pair
        else:
          continue
      else:
        unique_facts[fwd_dir] = (rel, score)
    unique_facts = list(unique_facts.items())
    return unique_facts

  def get_random_facts(self, entities, topk, alpha, seed_weighting=True):
      """Get random subgraph

      Args:
        entities: A list of Wikidata entities
        topk: Max entities to extract from PPR
        alpha: Node probability for PPR
        seed_weighting: Boolean for performing weighting seeds by freq in passage

      Returns:
        unique_facts: A list of unique random facts around the seeds.
      """
      #ent_ids = list(self.data.entity_names['e'].keys())
      ent_ids = [i for i in range(self.data.adj_mat_t_csr.shape[0])]
      extracted_ents = random.sample(ent_ids, 500)  # This doesn't work :(
      freq_dict = {}
      for i in extracted_ents:
          freq_dict[i] = 1
      extracted_scores = [1]*len(extracted_ents)
      facts = csr_topk_fact_extractor(self.data.adj_mat_t_csr, self.data.rel_dict,
                                      freq_dict, self.data.entity_names,
                                      extracted_ents, extracted_scores)
      if FLAGS.verbose_logging:
          tf.logging.info('Extracted facts: ')
          tf.logging.info(str(facts))

          # Extract 1 unique fact per pair of entities (fact with highest score)
          # Sort by scores
      unique_facts = {}
      for (sub, obj, rel, score) in facts:
          fwd_dir = (sub, obj)
          rev_dir = (obj, sub)
          if fwd_dir in unique_facts and score > unique_facts[fwd_dir][1]:
              unique_facts[fwd_dir] = (rel, score)
          elif rev_dir in unique_facts and score > unique_facts[rev_dir][1]:
              unique_facts[fwd_dir] = (rel, score)
              del unique_facts[rev_dir]  # Remove existing entity pair
          else:
              unique_facts[(sub, obj)] = (rel, score)
      unique_facts = list(unique_facts.items())
      return unique_facts
