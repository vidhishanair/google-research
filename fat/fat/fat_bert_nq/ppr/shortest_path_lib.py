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
from fat.fat_bert_nq.ppr.apr_algo import csr_personalized_pagerank, csr_get_shortest_path, csr_get_all_paths
from fat.fat_bert_nq.ppr.apr_algo import csr_topk_fact_extractor
from fat.fat_bert_nq.ppr.apr_algo import csr_get_k_hop_entities
from fat.fat_bert_nq.ppr.kb_csr_io import CsrData

flags = tf.flags
FLAGS = flags.FLAGS

#flags.DEFINE_bool(
#    'verbose_logging', False,
#    'If true, all of the warnings related to data processing will be printed. '
#    'A number of warnings are expected for a normal NQ evaluation.')

flags.DEFINE_integer(
    "k_hop", 2,
    "Num of hops for shortest path query")



class ShortestPath(object):
    """Shortest Path main lib which is used to wrap functions around Shortest Path algo."""

    def __init__(self, mode=None, task_id=None, shard_id=None, question_id=None):
        self.data = CsrData()
        self.data.load_csr_data(
            full_wiki=FLAGS.full_wiki, files_dir=FLAGS.apr_files_dir,
            mode=mode, task_id=task_id, shard_id=shard_id, question_id=question_id)
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

    def get_augmented_facts(self, path, entity_names, augmentation_type=None):
        augmented_path = []
        for single_path in path:
            single_path = single_path[1:]
            for (obj_id, rel_id, subj_id) in reversed(single_path):
                if obj_id == subj_id:
                    continue
                subj_name = entity_names['e'][str(subj_id)]['name']
                obj_name = entity_names['e'][str(obj_id)]['name'] if str(obj_id) != 'None' else 'None'
                rel_name = entity_names['r'][str(rel_id)]['name'] if str(rel_id) != 'None' else 'None'
                augmented_path.append((((subj_id, subj_name), (obj_id, obj_name)),
                                       ((rel_id, rel_name), None)))
        return augmented_path

    def get_all_path_augmented_facts(self, path, entity_names, augmentation_type=None):
        augmented_path = []
        for single_path in path:
            tmp_path = []
            single_path = single_path[1:]
            for (obj_id, rel_id, subj_id) in reversed(single_path):
                if obj_id == subj_id:
                    continue
                subj_name = entity_names['e'][str(subj_id)]['name']
                obj_name = entity_names['e'][str(obj_id)]['name'] if str(obj_id) != 'None' else 'None'
                rel_name = entity_names['r'][str(rel_id)]['name'] if str(rel_id) != 'None' else 'None'
                tmp_path.append((((subj_id, subj_name), (obj_id, obj_name)),
                                       ((rel_id, rel_name), None)))
            augmented_path.append(tmp_path)
        return augmented_path


    def get_shortest_path_facts(self, question_entities, answer_entities, passage_entities, seed_weighting=True, fp=None):
        """Get subgraph describing shortest path from question to answer.

        Args:
          question_entities: A list of Wikidata entities
          answer_entities: A list of Wikidata entities
          passage_entities: A list of Wikidata entities

        Returns:
          unique_facts: A list of unique facts representing the shortest path.
        """

        if FLAGS.verbose_logging:
            print('Getting subgraph')
            tf.logging.info('Getting subgraph')
        question_entity_ids = [
            int(self.data.ent2id[x]) for x in question_entities if x in self.data.ent2id
        ]
        question_entity_names = str([self.data.entity_names['e'][str(x)]['name'] for x in question_entity_ids
                                     ])
        #if fp is not None:
        #    fp.write(str(question_entities)+"\t"+question_entity_names+"\t")
        if FLAGS.verbose_logging:
            print('Question Entities')
            tf.logging.info('Question Entities')
            print(question_entities)
            print(question_entity_names)
            tf.logging.info(question_entity_names)

        answer_entity_ids = [
                int(self.data.ent2id[x]) for x in answer_entities if x in self.data.ent2id
            ]
        answer_entity_names = str([self.data.entity_names['e'][str(x)]['name'] for x in answer_entity_ids
                                   ])
        #if fp is not None:
        #    fp.write(str(answer_entities)+"\t"+answer_entity_names+"\t")
        if FLAGS.verbose_logging:
            print('Answer Entities')
            tf.logging.info('Answer Entities')
            print(answer_entities)
            print(answer_entity_names)
            tf.logging.info(answer_entity_names)
        passage_entity_ids = [
            int(self.data.ent2id[x]) for x in passage_entities if x in self.data.ent2id
        ]
        passage_entity_names = str([self.data.entity_names['e'][str(x)]['name'] for x in passage_entity_ids
                                   ])
        if FLAGS.verbose_logging:
            print('Passage Entities')
            tf.logging.info('Passage Entities')
            print(passage_entity_names)
            tf.logging.info(passage_entity_names)

        freq_dict = {x: question_entity_ids.count(x) for x in question_entity_ids}

        extracted_paths, num_hops = csr_get_shortest_path(question_entity_ids, self.data.adj_mat_t_csr, answer_entity_ids, self.data.rel_dict, k_hop=FLAGS.k_hop)
        augmented_facts = self.get_augmented_facts(extracted_paths, self.data.entity_names)

        if FLAGS.verbose_logging:
            print('Extracted facts: ')
            print(str(augmented_facts))
            tf.logging.info('Extracted facts: ')
            tf.logging.info(str(augmented_facts))
            print("Num hops: "+str(num_hops))
        return augmented_facts, num_hops

    def get_question_to_passage_facts(self, question_entities, answer_entities, passage_entities, seed_weighting=True, fp=None):
        """Get subgraph describing shortest path from question to answer.

        Args:
          question_entities: A list of Wikidata entities
          answer_entities: A list of Wikidata entities
          passage_entities: A list of Wikidata entities

        Returns:
          unique_facts: A list of unique facts representing the shortest path.
        """

        if FLAGS.verbose_logging:
            print('Getting subgraph')
            tf.logging.info('Getting subgraph')
        question_entity_ids = [
            int(self.data.ent2id[x]) for x in question_entities if x in self.data.ent2id
        ]
        question_entity_names = str([self.data.entity_names['e'][str(x)]['name'] for x in question_entity_ids
                                     ])
        #if fp is not None:
        #    fp.write(str(question_entities)+"\t"+question_entity_names+"\t")
        if FLAGS.verbose_logging:
            print('Question Entities')
            tf.logging.info('Question Entities')
            print(question_entities)
            print(question_entity_names)
            tf.logging.info(question_entity_names)

        answer_entity_ids = [
            int(self.data.ent2id[x]) for x in answer_entities if x in self.data.ent2id
        ]
        answer_entity_names = str([self.data.entity_names['e'][str(x)]['name'] for x in answer_entity_ids
                                   ])
        #if fp is not None:
        #    fp.write(str(answer_entities)+"\t"+answer_entity_names+"\t")
        if FLAGS.verbose_logging:
            print('Answer Entities')
            tf.logging.info('Answer Entities')
            print(answer_entities)
            print(answer_entity_names)
            tf.logging.info(answer_entity_names)
        passage_entity_ids = [
            int(self.data.ent2id[x]) for x in passage_entities if x in self.data.ent2id
        ]
        passage_entity_names = str([self.data.entity_names['e'][str(x)]['name'] for x in passage_entity_ids
                                    ])
        if FLAGS.verbose_logging:
            print('Passage Entities')
            tf.logging.info('Passage Entities')
            print(passage_entity_names)
            tf.logging.info(passage_entity_names)

        freq_dict = {x: question_entity_ids.count(x) for x in question_entity_ids}

        extracted_paths, num_hops = csr_get_all_paths(question_entity_ids, self.data.adj_mat_t_csr, passage_entity_ids, self.data.rel_dict, k_hop=FLAGS.k_hop)
        augmented_facts = self.get_all_path_augmented_facts(extracted_paths, self.data.entity_names)

        if FLAGS.verbose_logging:
            print('All path Extracted facts: ')
            print(str(augmented_facts))
            tf.logging.info('All path Extracted facts: ')
            tf.logging.info(str(augmented_facts))
            print("Num hops: "+str(num_hops))
        return augmented_facts, num_hops

    def get_all_path_facts(self, question_entities, answer_entities, passage_entities, seed_weighting=True, fp=None):
        """Get subgraph describing shortest path from question to answer.

        Args:
          question_entities: A list of Wikidata entities
          answer_entities: A list of Wikidata entities
          passage_entities: A list of Wikidata entities

        Returns:
          unique_facts: A list of unique facts representing the shortest path.
        """

        if FLAGS.verbose_logging:
            print('Getting subgraph')
            tf.logging.info('Getting subgraph')
        question_entity_ids = [
            int(self.data.ent2id[x]) for x in question_entities if x in self.data.ent2id
        ]
        question_entity_names = str([self.data.entity_names['e'][str(x)]['name'] for x in question_entity_ids
                                     ])
        #if fp is not None:
        #    fp.write(str(question_entities)+"\t"+question_entity_names+"\t")
        if FLAGS.verbose_logging:
            print('Question Entities')
            tf.logging.info('Question Entities')
            print(question_entities)
            print(question_entity_names)
            tf.logging.info(question_entity_names)

        answer_entity_ids = [
            int(self.data.ent2id[x]) for x in answer_entities if x in self.data.ent2id
        ]
        answer_entity_names = str([self.data.entity_names['e'][str(x)]['name'] for x in answer_entity_ids
                                   ])
        #if fp is not None:
        #    fp.write(str(answer_entities)+"\t"+answer_entity_names+"\t")
        if FLAGS.verbose_logging:
            print('Answer Entities')
            tf.logging.info('Answer Entities')
            print(answer_entities)
            print(answer_entity_names)
            tf.logging.info(answer_entity_names)
        passage_entity_ids = [
            int(self.data.ent2id[x]) for x in passage_entities if x in self.data.ent2id
        ]
        passage_entity_names = str([self.data.entity_names['e'][str(x)]['name'] for x in passage_entity_ids
                                    ])
        if FLAGS.verbose_logging:
            print('Passage Entities')
            tf.logging.info('Passage Entities')
            print(passage_entity_names)
            tf.logging.info(passage_entity_names)

        freq_dict = {x: question_entity_ids.count(x) for x in question_entity_ids}

        extracted_paths, num_hops = csr_get_all_paths(question_entity_ids, self.data.adj_mat_t_csr, answer_entity_ids, self.data.rel_dict, k_hop=FLAGS.k_hop)
        augmented_facts = self.get_all_path_augmented_facts(extracted_paths, self.data.entity_names)

        if FLAGS.verbose_logging:
            print('All path Extracted facts: ')
            print(str(augmented_facts))
            tf.logging.info('All path Extracted facts: ')
            tf.logging.info(str(augmented_facts))
            print("Num hops: "+str(num_hops))
        return augmented_facts, num_hops


if __name__ == '__main__':
    csr_data = ShortestPath(mode='train', task_id=0, shard_id=1)
    question_entities = ['Q3232520', 'Q932586']
    answer_entities = ['Q56146']
    #question_entities = ['Q954184', 'Q1046088']
    #answer_entities = ['Q869161']
    csr_data.get_shortest_path_facts(question_entities=question_entities, answer_entities=answer_entities,
                                     passage_entities=[], seed_weighting=False)
