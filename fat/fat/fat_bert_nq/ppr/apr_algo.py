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

"""This file holds the algorithm for the CSR based implementation os Personalized Page Rank."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'fact_score_type', 'FREQ_SCORE',
    'Scoring method for facts. One in ["FREQ_SCORE", "MIN_SCORE"]')


def csr_personalized_pagerank(seeds, adj_mat, alpha, max_iter=20):
  """Return the PPR Scores vector for the given seed and adjacency matrix.

  Algorithm :
      https://pdfs.semanticscholar.org/a4df/5ff749d823905ff9c1a23b522d3f426a1bb6.pdf
      (Figure 1)

  Args:
    seeds: A sparse matrix of size E x 1.
    adj_mat: A sparse matrix of size E x E whose rows sum to one.
    alpha: Probability of staying at current node [0-1]
    max_iter: Maximum iterations to run ppr for

  Returns:
    s_ovr: A vector of size E, ppr scores for every entity
  """

  restart_prob = alpha
  r = restart_prob * seeds
  s_ovr = np.copy(r)
  for _ in range(max_iter):
    if FLAGS.verbose_logging:
      tf.logging.info('Performing PPR Matrix Multiplication')
    st = time.time()
    r_new = (1. - restart_prob) * (adj_mat.dot(r))
    #print('Time taken for dot product: '+str(time.time() - st))
    s_ovr = s_ovr + r_new
    delta = abs(r_new.sum())
    if delta < 1e-5:
      break
    r = r_new
  return np.squeeze(s_ovr)


def csr_get_k_hop_entities(seeds, adj_mat, k_hop):
  """Return entities within k hop distance.

  Args:
    seeds: A list of seed entity ids
    adj_mat: A sparse matrix of size E x E whose rows sum to one.
    k_hop: No: of hops to extract entities from

  Returns:
      facts: A list of entities within k hop distance
  """
  k_hop_entities = seeds
  for i in range(k_hop):
    # Slicing adjacency matrix to subgraph of all extracted entities
    #print(seeds)
    submat = adj_mat[:, seeds]

    # Extracting non-zero entity pairs
    row, col = submat.nonzero()
    objects = []
    for ii in range(row.shape[0]):
      obj_id = row[ii]
      objects.append(obj_id)
    objects = list(set(objects))
    seeds = objects
    k_hop_entities.extend(objects)
  return k_hop_entities

def csr_get_shortest_path(question_seeds, adj_mat, answer_seeds, rel_dict, k_hop):
  """Return list of shortest paths between question and answer seeds.

  Args:
    question_seeds: A list of seed entity ids
    adj_mat: A sparse matrix of size E x E whose rows sum to one.
    answer_seeds: A list of seed entity ids

  Returns:
      paths: A list of shortest paths between question and answer entities
  """

  seeds = question_seeds
  answer_seeds = set(answer_seeds)
  parent_dict = {}
  answer_seeds_found = []
  for i in range(k_hop):
    # Slicing adjacency matrix to subgraph of all extracted entities
    print(seeds)
    submat = adj_mat[:, seeds]

    # Extracting non-zero entity pairs
    row, col = submat.nonzero()
    print(row)
    print(col)
    objects = []
    for ii in range(row.shape[0]):
      obj_id = row[ii]
      subj_id = seeds[col[ii]]
      # print('Processing link: '+str(subj_id)+" "+str(obj_id))
      if obj_id in parent_dict:
        if k_hop == parent_dict[obj_id][-1][1]:
          parent_dict[obj_id].append((subj_id, k_hop))
        else:
          parent_dict[obj_id] = [(subj_id, k_hop)]
      else:
        parent_dict[obj_id] = [(subj_id, k_hop)]
      objects.append(obj_id)
    objects = set(objects)
    answer_seeds_found = list(objects.intersection(answer_seeds))
    if answer_seeds_found:
      break
    seeds = list(objects)
  
  print('Answer seeds found' +str(answer_seeds_found))
  num_hops = i+1
  path = []
  for object in answer_seeds_found:
    path.append([(None, None, object)])

  for hop in range(num_hops):
    for i in range(len(path)):
      object = path[i][-1][2]
      for parent in parent_dict[object] : 
          parent = parent[0]
          rel = rel_dict[(parent, object)]
          path[i].append((object, rel, parent))
  print(path)
  return path

def get_augmented_facts(path, entity_names, augmentation_type='None'):
  augmented_path = []
  for single_path in path:
    augmented_path.append([])
    for (obj_id, rel_id, subj_id) in single_path[1:]:
      subj_name = entity_names['e'][str(subj_id)]['name']
      obj_name = entity_names['e'][str(obj_id)]['name'] if str(obj_id) != 'None' else 'None'
      rel_name = entity_names['r'][str(rel_id)]['name'] if str(rel_id) != 'None' else 'None'
      augmented_path[-1].append(((subj_id, subj_name),
                    (obj_id, obj_name),
                    (rel_id, rel_name), None))
  return augmented_path
def get_fact_score(extracted_scores,
                   subj,
                   obj,
                   freq_dict,
                   score_type='FREQ_SCORE'):
  """Return score for a subj, obj pair of entities.

  Args:
    extracted_scores: A score vector of size E
    subj: subj entity id
    obj: obj entity id
    freq_dict: frequency of every entity in passage
    score_type: string for type of scoring used

  Returns:
      score: A float score for a subj, obj entity pair
  """
  score_types = set(['FREQ_SCORE', 'MIN_SCORE'])
  # Min of Page Rank scores of both Entities
  # Upweight facts where both have high scores
  min_score = min(
      extracted_scores[subj], extracted_scores[obj]
  )

  # Freq Score - If both entities are present - sum of frequencies
  # Upweight facts where both entities are in passage
  if subj in freq_dict and obj in freq_dict:
    freq_score = freq_dict[subj] + freq_dict[obj]
  else:
    freq_score = min(extracted_scores[subj],
                     extracted_scores[obj])
  if score_type == 'FREQ_SCORE':
    return freq_score
  elif score_type == 'MIN_SCORE':
    return min_score
  else:
    ValueError(
        'The score_type should be one of: %s' + ', '.join(list(score_types)))


def csr_topk_fact_extractor(adj_mat, rel_dict, freq_dict, entity_names,
                            extracted_ents, extracted_scores):
  """Return facts for selected entities.

  Args:
    adj_mat: A sparse matrix of size E x E whose rows sum to one.
    rel_dict: A sparse matrix of size E x E whose values are rel_ids between
          entities
    freq_dict: A dictionary with frequency of every entity in passage
    entity_names: A dictionary of entity and relation ids to their surface
          form names
    extracted_ents: A list of selected topk entities
    extracted_scores: A list of selected topk entity scores

  Returns:
      facts: A list of ((subj_id, subj_name), (obj_id, obj_name), (rel_id,
          rel_name), score)
  """

  # Slicing adjacency matrix to subgraph of all extracted entities
  submat = adj_mat[extracted_ents, :]
  submat = submat[:, extracted_ents]
  # Extracting non-zero entity pairs
  col_idx, row_idx = submat.nonzero()

  facts = []
  for ii in range(row_idx.shape[0]):
    subj_id = extracted_ents[row_idx[ii]]
    obj_id = extracted_ents[col_idx[ii]]
    fwd_dir = (subj_id, obj_id)
    rev_dir = (obj_id, subj_id)
    rel_id = rel_dict[fwd_dir]
    if rel_id == 0:  # no relation from subj to obj
      # Checking for relation from obj to subj
      rel_id = rel_dict[rev_dir]
      if rel_id == 0:
        continue
      subj_id, obj_id = obj_id, subj_id
    score = get_fact_score(
        extracted_scores,
        row_idx[ii],
        col_idx[ii],
        freq_dict,
        score_type=FLAGS.fact_score_type)
    subj_name = entity_names['e'][str(subj_id)]['name']
    obj_name = entity_names['e'][str(obj_id)]['name']
    rel_name = entity_names['r'][str(rel_id)]['name']
    facts.append(((subj_id, subj_name),
                  (obj_id, obj_name),
                  (rel_id, rel_name), score))
  return facts


# def csr_topk_fact_extractor(adj_mat, rel_dict, entity_names):
#     """Return random facts.
#
#     Args:
#       adj_mat: A sparse matrix of size E x E whose rows sum to one.
#       rel_dict: A sparse matrix of size E x E whose values are rel_ids between
#             entities
#       entity_names: A dictionary of entity and relation ids to their surface
#             form names
#
#
#     Returns:
#         facts: A list of ((subj_id, subj_name), (obj_id, obj_name), (rel_id,
#             rel_name), score)
#     """
#
