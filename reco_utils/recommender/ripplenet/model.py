# This code is modified from RippleNet
# Online code of RippleNet: https://github.com/hwwang55/RippleNet

import tensorflow as tf
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class RippleNet(object):
    """RippleNet Implementation. RippleNet is an end-to-end framework that naturally
    incorporates the knowledge graphs into recommender systems.
    Similar to actual ripples propagating on the water, RippleNet stimulates the propagation
    of user preferences over the set of knowledge entities by automatically and iteratively
    extending a user’s potential interests along links in the knowledge graph.
    """

    def __init__(
        self,
        dim,
        n_hop,
        kge_weight,
        l2_weight,
        lr,
        n_memory,
        item_update_mode,
        using_all_hops,
        n_entity,
        n_relation,
        optimizer_method="adam",
        seed=None,
    ):

        """Initialize model parameters

        Args:
            dim (int): dimension of entity and relation embeddings
            n_hop (int): maximum hops to create ripples using the KG
            kge_weight (float): weight of the KGE term
            l2_weight (float): weight of the l2 regularization term
            lr (float): learning rate
            n_memory (int): size of ripple set for each hop
            item_update_mode (string): how to update item at the end of each hop. 
                                    possible options are replace, plus, plus_transform or replace transform
            using_all_hops (bool): whether to use outputs of all hops or just the
                                   last hop when making prediction
            n_entity (int): number of entitites in the KG
            n_relation (int): number of types of relations in the KG
            optimizer_method (string): optimizer method from adam, adadelta, adagrad, ftrl (FtrlOptimizer),
                          #gd (GradientDescentOptimizer), rmsprop (RMSPropOptimizer)
            seed (int): initial seed value
        """
        self.seed = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = dim
        self.n_hop = n_hop
        self.kge_weight = kge_weight
        self.l2_weight = l2_weight
        self.lr = lr
        self.n_memory = n_memory
        self.item_update_mode = item_update_mode
        self.using_all_hops = using_all_hops
        self.optimizer_method = optimizer_method

        self._build_inputs()
        self._build_embeddings()
        self._build_model()
        self._build_loss()
        self._build_optimizer()

        self.init_op = tf.global_variables_initializer()

        # set GPU use with demand growth
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init_op)

    def _build_inputs(self):
        self.items = tf.placeholder(dtype=tf.int32, shape=[None], name="items")
        self.labels = tf.placeholder(dtype=tf.float64, shape=[None], name="labels")
        self.memories_h = []
        self.memories_r = []
        self.memories_t = []

        for hop in range(self.n_hop):
            self.memories_h.append(
                tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, self.n_memory],
                    name="memories_h_" + str(hop),
                )
            )
            self.memories_r.append(
                tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, self.n_memory],
                    name="memories_r_" + str(hop),
                )
            )
            self.memories_t.append(
                tf.placeholder(
                    dtype=tf.int32,
                    shape=[None, self.n_memory],
                    name="memories_t_" + str(hop),
                )
            )

    def _build_embeddings(self):
        self.entity_emb_matrix = tf.get_variable(
            name="entity_emb_matrix",
            dtype=tf.float64,
            shape=[self.n_entity, self.dim],
            initializer=tf.contrib.layers.xavier_initializer(),
        )
        self.relation_emb_matrix = tf.get_variable(
            name="relation_emb_matrix",
            dtype=tf.float64,
            shape=[self.n_relation, self.dim, self.dim],
            initializer=tf.contrib.layers.xavier_initializer(),
        )

    def _build_model(self):
        # transformation matrix for updating item embeddings at the end of each hop
        self.transform_matrix = tf.get_variable(
            name="transform_matrix",
            shape=[self.dim, self.dim],
            dtype=tf.float64,
            initializer=tf.contrib.layers.xavier_initializer(),
        )

        # [batch size, dim]
        self.item_embeddings = tf.nn.embedding_lookup(
            self.entity_emb_matrix, self.items
        )

        self.h_emb_list = []
        self.r_emb_list = []
        self.t_emb_list = []
        for i in range(self.n_hop):
            # [batch size, n_memory, dim]
            self.h_emb_list.append(
                tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_h[i])
            )

            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(
                tf.nn.embedding_lookup(self.relation_emb_matrix, self.memories_r[i])
            )

            # [batch size, n_memory, dim]
            self.t_emb_list.append(
                tf.nn.embedding_lookup(self.entity_emb_matrix, self.memories_t[i])
            )

        o_list = self._key_addressing()

        self.scores = tf.squeeze(self._predict_scores(self.item_embeddings, o_list))
        self.scores_normalized = tf.sigmoid(self.scores)

    def _key_addressing(self):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=3)

            # [batch_size, n_memory, dim]
            Rh = tf.squeeze(tf.matmul(self.r_emb_list[hop], h_expanded), axis=3)

            # [batch_size, dim, 1]
            v = tf.expand_dims(self.item_embeddings, axis=2)

            # [batch_size, n_memory]
            probs = tf.squeeze(tf.matmul(Rh, v), axis=2)

            # [batch_size, n_memory]
            probs_normalized = tf.nn.softmax(probs)

            # [batch_size, n_memory, 1]
            probs_expanded = tf.expand_dims(probs_normalized, axis=2)

            # [batch_size, dim]
            o = tf.reduce_sum(self.t_emb_list[hop] * probs_expanded, axis=1)

            self.item_embeddings = self._update_item_embedding(self.item_embeddings, o)
            o_list.append(o)
        return o_list

    def _update_item_embedding(self, item_embeddings, o):

        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = tf.matmul(o, self.transform_matrix)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = tf.matmul(item_embeddings + o, self.transform_matrix)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings

    def _predict_scores(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]

        scores = tf.reduce_sum(item_embeddings * y, axis=1)
        return scores

    def _build_loss(self):
        self.base_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self.labels, logits=self.scores
            )
        )

        self.kge_loss = 0
        for hop in range(self.n_hop):
            h_expanded = tf.expand_dims(self.h_emb_list[hop], axis=2)
            t_expanded = tf.expand_dims(self.t_emb_list[hop], axis=3)
            hRt = tf.squeeze(
                tf.matmul(tf.matmul(h_expanded, self.r_emb_list[hop]), t_expanded)
            )
            self.kge_loss += tf.reduce_mean(tf.sigmoid(hRt))
        self.kge_loss = -self.kge_weight * self.kge_loss

        self.l2_loss = 0
        for hop in range(self.n_hop):
            self.l2_loss += tf.reduce_mean(
                tf.reduce_sum(self.h_emb_list[hop] * self.h_emb_list[hop])
            )
            self.l2_loss += tf.reduce_mean(
                tf.reduce_sum(self.t_emb_list[hop] * self.t_emb_list[hop])
            )
            self.l2_loss += tf.reduce_mean(
                tf.reduce_sum(self.r_emb_list[hop] * self.r_emb_list[hop])
            )
            if (
                self.item_update_mode == "replace nonlinear"
                or self.item_update_mode == "plus nonlinear"
            ):
                self.l2_loss += tf.nn.l2_loss(self.transform_matrix)
        self.l2_loss = self.l2_weight * self.l2_loss

        self.loss = self.base_loss + self.kge_loss + self.l2_loss

    def _build_optimizer(self):

        if self.optimizer_method == "adam":
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        elif self.optimizer_method == "adadelta":
            self.optimizer = tf.train.AdadeltaOptimizer(self.lr).minimize(self.loss)
        elif self.optimizer_method == "adagrad":
            self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.loss)
        elif self.optimizer_method == "ftrl":
            self.optimizer = tf.train.FtrlOptimizer(self.lr).minimize(self.loss)
        elif self.optimizer_method == "gd":
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(
                self.loss
            )
        elif self.optimizer_method == "rmsprop":
            self.optimizer = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        else:
            raise Exception("Unkown optimizer method: " + self.optimizer_method)

    def _train(self, feed_dict):
        return self.sess.run([self.optimizer, self.loss], feed_dict)

    def _return_scores(self, feed_dict):
        labels, scores = self.sess.run([self.labels, self.scores_normalized], feed_dict)
        return labels, scores

    def _eval(self, feed_dict):
        labels, scores = self.sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc

    def _get_feed_dict(self, data, start, end):
        feed_dict = dict()
        feed_dict[self.items] = data[start:end, 1]
        feed_dict[self.labels] = data[start:end, 2]
        for i in range(self.n_hop):
            feed_dict[self.memories_h[i]] = [
                self.ripple_set[user][i][0] for user in data[start:end, 0]
            ]
            feed_dict[self.memories_r[i]] = [
                self.ripple_set[user][i][1] for user in data[start:end, 0]
            ]
            feed_dict[self.memories_t[i]] = [
                self.ripple_set[user][i][2] for user in data[start:end, 0]
            ]
        return feed_dict

    def _print_metrics_evaluation(self, data, batch_size):
        start = 0
        auc_list = []
        acc_list = []
        while start < data.shape[0]:
            auc, acc = self._eval(
                self._get_feed_dict(data=data, start=start, end=start + batch_size)
            )
            auc_list.append(auc)
            acc_list.append(acc)
            start += batch_size
        return float(np.mean(auc_list)), float(np.mean(acc_list))

    def fit(self, n_epoch, batch_size, train_data, ripple_set, show_loss):
        """Main fit method for RippleNet.

        Args:
            n_epoch (int): the number of epochs
            batch_size (int): batch size
            train_data (pd.DataFrame): User id, item and rating dataframe
            ripple_set (dictionary): set of knowledge triples per user positive rating, from 0 until n_hop
            show_loss (bool): whether to show loss update
        """
        self.ripple_set = ripple_set
        self.train_data = train_data.values
        for step in range(n_epoch):
            # training
            np.random.shuffle(self.train_data)
            start = 0
            while start < self.train_data.shape[0]:
                _, loss = self._train(
                    self._get_feed_dict(
                        data=self.train_data, start=start, end=start + batch_size
                    )
                )
                start += batch_size
                if show_loss:
                    log.info(
                        "%.1f%% %.4f" % (start / self.train_data.shape[0] * 100, loss)
                    )

            train_auc, train_acc = self._print_metrics_evaluation(
                data=self.train_data, batch_size=batch_size
            )

            log.info(
                "epoch %d  train auc: %.4f  acc: %.4f" % (step, train_auc, train_acc)
            )

    def predict(self, batch_size, data):
        """Main predict method for RippleNet.

        Args:
            batch_size (int): batch size
            data (pd.DataFrame): User id, item and rating dataframe
        
        Returns:
            (list, list): real labels of the predicted items, predicted scores of the predicted items
        """
        data = data.values
        start = 0
        labels = [0] * data.shape[0]
        scores = [0] * data.shape[0]
        while start < data.shape[0]:
            (
                labels[start : start + batch_size],
                scores[start : start + batch_size],
            ) = self._return_scores(
                feed_dict=self._get_feed_dict(
                    data=data, start=start, end=start + batch_size
                )
            )
            start += batch_size

        return labels, scores

    def recommend_k_items(self, batch_size, data, top_k=10, remove_seen=True):
        """Recommend top K items method for RippleNet.

        Args:
            batch_size (int): batch size
            data (pd.DataFrame): User id, item and rating dataframe
            top_k (int): number of items to recommend
            remove_seen (bool): if the items seen by an user in train should be recomed from the test set
        
        Returns:
            (pd.DataFrame): top K items by score per user
        """
        if remove_seen == True:
            log.info("Removing seen items")
            train_data = pd.DataFrame(self.train_data).iloc[:, 0:2]
            train_data.columns = list(data.columns[0:2])
            seen_items = data.merge(
                train_data.iloc[:, 0:2],
                on=list(data.columns[0:2]),
                indicator=True,
                how="left",
            )
            data = seen_items[seen_items["_merge"] == "left_only"].drop(
                columns=["_merge"]
            )
        data_np = data.values
        start = 0
        labels = [0] * data_np.shape[0]
        scores = [0] * data_np.shape[0]
        while start < data_np.shape[0]:
            (
                labels[start : start + batch_size],
                scores[start : start + batch_size],
            ) = self._return_scores(
                feed_dict=self._get_feed_dict(
                    data=data_np, start=start, end=start + batch_size
                )
            )
            start += batch_size

        data["scores"] = scores
        top_k_items = (
            data.groupby(data.columns[0], as_index=False)
            .apply(lambda x: x.nlargest(top_k, "scores"))
            .reset_index(drop=True)
        )
        # Add ranks
        top_k_items["rank"] = (
            top_k_items.groupby(data.columns[0], sort=False).cumcount() + 1
        )

        return top_k_items
