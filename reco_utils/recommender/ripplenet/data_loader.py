# This code is modified from RippleNet
# Online code of RippleNet: https://github.com/hwwang55/RippleNet

import collections
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_kg(kg_final):
    """Standarize indexes for items and entities

    Args:
        kg_final (pd.DataFrame): knowledge graph converted with columns head,
         relation and tail, with internal entity IDs

    Returns:
        n_entity (int): number of entities in KG
        n_relation (int): number of relations in KG
        kg (dictionary): KG in dictionary shape
    """
    log.info("reading KG file ...")

    n_entity = len(set(kg_final.iloc[:, 0]) | set(kg_final.iloc[:, 2]))
    n_relation = len(set(kg_final.iloc[:, 1]))

    kg = collections.defaultdict(list)
    for index, row in kg_final.iterrows():
        kg[row["head"]].append((row["tail"], row["relation"]))

    return n_entity, n_relation, kg


def get_ripple_set(kg, user_history_dict, n_hop=2, n_memory=36):
    """Build Ripple Set, dictionary for the related entities in the KG
     given the paths of users, number of hops and memory

    Args:
        kg (dictionary): KG in dictionary shape
        user_history_dict (dictionary): positive ratings from train data, to build ripple structure
        n_hop (int): int, maximum hops in the KG
        n_memory (int): int, size of ripple set for each hop

    Returns:
        ripple_set (dictionary): set of knowledge triples per user positive rating, from 0 until n_hop
    """
    log.info("constructing ripple set ...")

    # user -> [(hop_0_heads, hop_0_relations, hop_0_tails), (hop_1_heads, hop_1_relations, hop_1_tails), ...]
    ripple_set = collections.defaultdict(list)

    for user in user_history_dict:
        for h in range(n_hop):
            memories_h = []
            memories_r = []
            memories_t = []

            if h == 0:
                tails_of_last_hop = user_history_dict[user]
            else:
                tails_of_last_hop = ripple_set[user][-1][2]

            for entity in tails_of_last_hop:
                for tail_and_relation in kg[entity]:
                    memories_h.append(entity)
                    memories_r.append(tail_and_relation[1])
                    memories_t.append(tail_and_relation[0])

            # if the current ripple set of the given user is empty, we simply copy the ripple set of the last hop here
            # this won't happen for h = 0, because only the items that appear in the KG have been selected
            # this only happens on 154 users in Book-Crossing dataset (since both BX dataset and the KG are sparse)
            if len(memories_h) == 0:
                ripple_set[user].append(ripple_set[user][-1])
            else:
                # sample a fixed-size 1-hop memory for each user
                replace = len(memories_h) < n_memory
                indices = np.random.choice(
                    len(memories_h), size=n_memory, replace=replace
                )
                memories_h = [memories_h[i] for i in indices]
                memories_r = [memories_r[i] for i in indices]
                memories_t = [memories_t[i] for i in indices]
                ripple_set[user].append((memories_h, memories_r, memories_t))

    return ripple_set
