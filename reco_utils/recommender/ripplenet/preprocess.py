# This code is modified from RippleNet
# Online code of RippleNet: https://github.com/hwwang55/RippleNet

import argparse
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def read_item_index_to_entity_id_file(item_to_entity):
    """Standarize indexes for items and entities

    Args:
        item_to_entity (pd.DataFrame): KG dataframe with original item and entity IDs

    Returns:
        item_index_old2new (dictionary): dictionary conversion from original item ID to internal item ID 
        entity_id2index (dictionary): dictionary conversion from original entity ID to internal entity ID 
    """
    item_index_old2new = dict()
    entity_id2index = dict()
    i = 0
    for index, row in item_to_entity.iterrows():
        item_index = str(row[0])
        satori_id = str(row[1])
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1
    return item_index_old2new, entity_id2index


def convert_rating(ratings, item_index_old2new, threshold, remove_negative_ratings=True, seed=14):
    """Apply item standarization to ratings dataset. 
    Use rating threshold to determite positive ratings

    Args:
        ratings (pd.DataFrame): ratings with columns ["UserId", "ItemId", "Rating"]
        item_index_old2new (dictionary): dictionary, conversion from original item ID to internal item ID
        threshold (int): minimum valur for the rating to be considered positive
        remove_negative_ratings (bool): if the train/test set should exclude items below the threshold,
                                        as the original papel proposes 

    Returns:
        ratings_final (pd.DataFrame): ratings converted with columns userID,
         internal item ID and binary rating (1, 0)
    """
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for index, row in ratings.iterrows():
        item_index_old = str(int(row[1]))
        if (
            item_index_old not in item_index_old2new
        ):  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(row[0])

        rating = float(row[2])
        if rating >= threshold:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            user_pos_ratings[user_index_old].add((item_index, rating))
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            user_neg_ratings[user_index_old].add((item_index, rating))

    log.info("converting rating file ...")
    writer = []
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]
        for item, original_rating in pos_item_set:
            writer.append(
                {
                    "user_index": user_index,
                    "item": item,
                    "rating": 1,
                    "original_rating": original_rating,
                }
            )
        pos_item_set = set(i[0] for i in pos_item_set)
        unwatched_set = item_set - pos_item_set
        if user_index_old in user_neg_ratings:
            negative_set = dict(list(user_neg_ratings[user_index_old]))
            if remove_negative_ratings == True:
                unwatched_set -= set(negative_set.keys())
        else:
            negative_set = {}
        np.random.seed(seed)
        for item in np.random.choice(
            list(unwatched_set), size=len(pos_item_set), replace=False
        ):
            if item in negative_set:
                original_rating = negative_set[item]
            else:
                original_rating = 0
            writer.append(
                {
                    "user_index": user_index,
                    "item": item,
                    "rating": 0,
                    "original_rating": original_rating,
                }
            )
    ratings_final = pd.DataFrame(writer)
    log.info("number of users: %d" % user_cnt)
    log.info("number of items: %d" % len(item_set))
    return ratings_final


def convert_kg(kg, entity_id2index):
    """Apply entity standarization to KG dataset
    Args:
        kg (pd.DataFrame): knowledge graph with columns ["original_entity_id", "relation", "linked_entities_id"]
        entity_id2index (pd.DataFrame): dictionary, conversion from original entity ID to internal entity ID

    Returns:
        kg_final (pd.DataFrame): knowledge graph converted with columns head,
         relation and tail, with internal entity IDs
    """
    log.info("converting kg file ...")
    entity_cnt = len(entity_id2index)
    relation_cnt = 0
    relation_id2index = dict()

    writer = []

    for index, row in kg.iterrows():
        head_old = str(int(row[0]))
        relation_old = row[1]
        tail_old = str(int(row[2]))

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.append({"head": head, "relation": relation, "tail": tail})

    kg_final = pd.DataFrame(writer)
    log.info("number of entities (containing items): %d" % entity_cnt)
    log.info("number of relations: %d" % relation_cnt)
    return kg_final
