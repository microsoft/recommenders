# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.
import os
os.environ['TF_USE_LEGACY_KERAS'] = "1"

import numpy as np
import tensorflow as tf

from recommenders.models.deeprec.models.base_model import BaseModel

__all__ = ["DKN"]


class DKN(BaseModel):
    """DKN model (Deep Knowledge-Aware Network)

    :Citation:

        H. Wang, F. Zhang, X. Xie and M. Guo, "DKN: Deep Knowledge-Aware Network for News
        Recommendation", in Proceedings of the 2018 World Wide Web Conference on World
        Wide Web, 2018.
    """

    def __init__(self, hparams, iterator_creator):
        """Initialization steps for DKN.
        Compared with the BaseModel, DKN requires two different pre-computed embeddings,
        i.e. word embedding and entity embedding.
        After creating these two embedding variables, BaseModel's `__init__` method will be called.

        Args:
            hparams (object): Global hyper-parameters.
            iterator_creator (object): DKN data loader class.
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.compat.v1.name_scope("embedding"):
                word2vec_embedding = self._init_embedding(hparams.wordEmb_file)
                self.embedding = tf.Variable(
                    word2vec_embedding, trainable=True, name="word"
                )

                if hparams.use_entity:
                    e_embedding = self._init_embedding(hparams.entityEmb_file)
                    W = tf.Variable(
                        tf.random.uniform([hparams.entity_dim, hparams.dim], -1, 1),
                        trainable=True,
                    )
                    b = tf.Variable(tf.zeros([hparams.dim]), trainable=True)
                    self.entity_embedding = tf.nn.tanh(tf.matmul(e_embedding, W) + b)
                else:
                    self.entity_embedding = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=[hparams.entity_size, hparams.dim],
                            dtype=tf.float32,
                        ),
                        trainable=True,
                        name="entity",
                    )

                if hparams.use_context:
                    c_embedding = self._init_embedding(hparams.contextEmb_file)
                    W = tf.Variable(
                        tf.random.uniform([hparams.entity_dim, hparams.dim], -1, 1),
                        trainable=True,
                    )
                    b = tf.Variable(tf.zeros([hparams.dim]), trainable=True)
                    self.context_embedding = tf.nn.tanh(tf.matmul(c_embedding, W) + b)
                else:
                    self.context_embedding = tf.Variable(
                        tf.constant(
                            0.0,
                            shape=[hparams.entity_size, hparams.dim],
                            dtype=tf.float32,
                        ),
                        trainable=True,
                        name="context",
                    )

        super().__init__(hparams, iterator_creator, graph=self.graph)

    def _init_embedding(self, file_path):
        """Load pre-trained embeddings as a constant tensor.

        Args:
            file_path (str): the pre-trained embeddings filename.

        Returns:
            object: A constant tensor.
        """
        return tf.constant(np.load(file_path).astype(np.float32))

    def _l2_loss(self):
        hparams = self.hparams
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        l2_loss = tf.add(
            l2_loss, tf.multiply(hparams.embed_l2, tf.nn.l2_loss(self.embedding))
        )
        if hparams.use_entity:
            l2_loss = tf.add(
                l2_loss,
                tf.multiply(hparams.embed_l2, tf.nn.l2_loss(self.entity_embedding)),
            )
        if hparams.use_entity and hparams.use_context:
            l2_loss = tf.add(
                l2_loss,
                tf.multiply(hparams.embed_l2, tf.nn.l2_loss(self.context_embedding)),
            )
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(hparams.layer_l2, tf.nn.l2_loss(param))
            )
        return l2_loss

    def _l1_loss(self):
        hparams = self.hparams
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        l1_loss = tf.add(
            l1_loss,
            tf.multiply(hparams.embed_l1, tf.norm(tensor=self.embedding, ord=1)),
        )
        if hparams.use_entity:
            l1_loss = tf.add(
                l1_loss,
                tf.multiply(
                    hparams.embed_l1, tf.norm(tensor=self.entity_embedding, ord=1)
                ),
            )
        if hparams.use_entity and hparams.use_context:
            l1_loss = tf.add(
                l1_loss,
                tf.multiply(
                    hparams.embed_l1, tf.norm(tensor=self.context_embedding, ord=1)
                ),
            )
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(hparams.layer_l1, tf.norm(tensor=param, ord=1))
            )
        return l1_loss

    def _build_graph(self):
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)
        with tf.compat.v1.variable_scope("DKN"):
            logit = self._build_dkn()
            return logit

    def _build_dkn(self):
        """The main function to create DKN's logic.

        Returns:
            object: Prediction score made by the DKN model.
        """
        hparams = self.hparams
        # build attention model for clicked news and candidate news
        click_news_embed_batch, candidate_news_embed_batch = self._build_pair_attention(
            self.iterator.candidate_news_index_batch,
            self.iterator.candidate_news_entity_index_batch,
            self.iterator.click_news_index_batch,
            self.iterator.click_news_entity_index_batch,
            hparams,
        )

        nn_input = tf.concat(
            [click_news_embed_batch, candidate_news_embed_batch], axis=1
        )

        dnn_channel_part = 2
        last_layer_size = dnn_channel_part * self.num_filters_total
        layer_idx = 0
        hidden_nn_layers = []
        hidden_nn_layers.append(nn_input)
        with tf.compat.v1.variable_scope("nn_part", initializer=self.initializer):
            for idx, layer_size in enumerate(hparams.layer_sizes):
                curr_w_nn_layer = tf.compat.v1.get_variable(
                    name="w_nn_layer" + str(layer_idx),
                    shape=[last_layer_size, layer_size],
                    dtype=tf.float32,
                )
                curr_b_nn_layer = tf.compat.v1.get_variable(
                    name="b_nn_layer" + str(layer_idx),
                    shape=[layer_size],
                    dtype=tf.float32,
                )
                curr_hidden_nn_layer = tf.compat.v1.nn.xw_plus_b(
                    hidden_nn_layers[layer_idx], curr_w_nn_layer, curr_b_nn_layer
                )
                if hparams.enable_BN is True:
                    curr_hidden_nn_layer = tf.compat.v1.layers.batch_normalization(
                        curr_hidden_nn_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                activation = hparams.activation[idx]
                curr_hidden_nn_layer = self._active_layer(
                    logit=curr_hidden_nn_layer, activation=activation
                )
                hidden_nn_layers.append(curr_hidden_nn_layer)
                layer_idx += 1
                last_layer_size = layer_size
                self.layer_params.append(curr_w_nn_layer)
                self.layer_params.append(curr_b_nn_layer)

            w_nn_output = tf.compat.v1.get_variable(
                name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
            )
            b_nn_output = tf.compat.v1.get_variable(
                name="b_nn_output", shape=[1], dtype=tf.float32
            )
            self.layer_params.append(w_nn_output)
            self.layer_params.append(b_nn_output)
            nn_output = tf.compat.v1.nn.xw_plus_b(
                hidden_nn_layers[-1], w_nn_output, b_nn_output
            )
            return nn_output

    def _build_pair_attention(
        self,
        candidate_word_batch,
        candidate_entity_batch,
        click_word_batch,
        click_entity_batch,
        hparams,
    ):
        """This function learns the candidate news article's embedding and user embedding.
        User embedding is generated from click history and also depends on the candidate news article via attention mechanism.
        Article embedding is generated via KCNN module.
        Args:
            candidate_word_batch (object): tensor word indices for constructing news article
            candidate_entity_batch (object): tensor entity values for constructing news article
            click_word_batch (object): tensor word indices for constructing user clicked history
            click_entity_batch (object): tensor entity indices for constructing user clicked history
            hparams (object): global hyper-parameters
        Returns:
            click_field_embed_final_batch: user embedding
            news_field_embed_final_batch: candidate news article embedding

        """
        doc_size = hparams.doc_size
        attention_hidden_sizes = hparams.attention_layer_sizes

        clicked_words = tf.reshape(click_word_batch, shape=[-1, doc_size])
        clicked_entities = tf.reshape(click_entity_batch, shape=[-1, doc_size])

        with tf.compat.v1.variable_scope(
            "attention_net", initializer=self.initializer
        ) as scope:  # noqa: F841

            # use kims cnn to get conv embedding
            with tf.compat.v1.variable_scope(
                "kcnn", initializer=self.initializer, reuse=tf.compat.v1.AUTO_REUSE
            ) as cnn_scope:  # noqa: F841
                news_field_embed = self._kims_cnn(
                    candidate_word_batch, candidate_entity_batch, hparams
                )
                click_field_embed = self._kims_cnn(
                    clicked_words, clicked_entities, hparams
                )
                click_field_embed = tf.reshape(
                    click_field_embed,
                    shape=[
                        -1,
                        hparams.history_size,
                        hparams.num_filters * len(hparams.filter_sizes),
                    ],
                )

            avg_strategy = False
            if avg_strategy:
                click_field_embed_final = tf.reduce_mean(
                    input_tensor=click_field_embed, axis=1, keepdims=True
                )
            else:
                news_field_embed = tf.expand_dims(news_field_embed, 1)
                news_field_embed_repeat = tf.add(
                    tf.zeros_like(click_field_embed), news_field_embed
                )
                attention_x = tf.concat(
                    axis=-1, values=[click_field_embed, news_field_embed_repeat]
                )
                attention_x = tf.reshape(
                    attention_x, shape=[-1, self.num_filters_total * 2]
                )
                attention_w = tf.compat.v1.get_variable(
                    name="attention_hidden_w",
                    shape=[self.num_filters_total * 2, attention_hidden_sizes],
                    dtype=tf.float32,
                )
                attention_b = tf.compat.v1.get_variable(
                    name="attention_hidden_b",
                    shape=[attention_hidden_sizes],
                    dtype=tf.float32,
                )
                curr_attention_layer = tf.compat.v1.nn.xw_plus_b(
                    attention_x, attention_w, attention_b
                )

                if hparams.enable_BN is True:
                    curr_attention_layer = tf.compat.v1.layers.batch_normalization(
                        curr_attention_layer,
                        momentum=0.95,
                        epsilon=0.0001,
                        training=self.is_train_stage,
                    )

                activation = hparams.attention_activation
                curr_attention_layer = self._active_layer(
                    logit=curr_attention_layer, activation=activation
                )
                attention_output_w = tf.compat.v1.get_variable(
                    name="attention_output_w",
                    shape=[attention_hidden_sizes, 1],
                    dtype=tf.float32,
                )
                attention_output_b = tf.compat.v1.get_variable(
                    name="attention_output_b", shape=[1], dtype=tf.float32
                )
                attention_weight = tf.compat.v1.nn.xw_plus_b(
                    curr_attention_layer, attention_output_w, attention_output_b
                )
                attention_weight = tf.reshape(
                    attention_weight, shape=[-1, hparams.history_size, 1]
                )
                norm_attention_weight = tf.nn.softmax(attention_weight, axis=1)
                click_field_embed_final = tf.reduce_sum(
                    input_tensor=tf.multiply(click_field_embed, norm_attention_weight),
                    axis=1,
                    keepdims=True,
                )
                if attention_w not in self.layer_params:
                    self.layer_params.append(attention_w)
                if attention_b not in self.layer_params:
                    self.layer_params.append(attention_b)
                if attention_output_w not in self.layer_params:
                    self.layer_params.append(attention_output_w)
                if attention_output_b not in self.layer_params:
                    self.layer_params.append(attention_output_b)
            self.news_field_embed_final_batch = tf.squeeze(news_field_embed)
            click_field_embed_final_batch = tf.squeeze(click_field_embed_final)

        return click_field_embed_final_batch, self.news_field_embed_final_batch

    def _kims_cnn(self, word, entity, hparams):
        """The KCNN module. KCNN is an extension of traditional CNN that incorporates symbolic knowledge from
        a knowledge graph into sentence representation learning.
        Args:
            word (object): word indices for the sentence.
            entity (object): entity indices for the sentence. Entities are aligned with words in the sentence.
            hparams (object): global hyper-parameters.

        Returns:
            object: Sentence representation.
        """
        # kims cnn parameter
        filter_sizes = hparams.filter_sizes
        num_filters = hparams.num_filters

        dim = hparams.dim
        embedded_chars = tf.nn.embedding_lookup(params=self.embedding, ids=word)
        if hparams.use_entity and hparams.use_context:
            entity_embedded_chars = tf.nn.embedding_lookup(
                params=self.entity_embedding, ids=entity
            )
            context_embedded_chars = tf.nn.embedding_lookup(
                params=self.context_embedding, ids=entity
            )
            concat = tf.concat(
                [embedded_chars, entity_embedded_chars, context_embedded_chars], axis=-1
            )
        elif hparams.use_entity:
            entity_embedded_chars = tf.nn.embedding_lookup(
                params=self.entity_embedding, ids=entity
            )
            concat = tf.concat([embedded_chars, entity_embedded_chars], axis=-1)
        else:
            concat = embedded_chars
        concat_expanded = tf.expand_dims(concat, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.compat.v1.variable_scope(
                "conv-maxpool-%s" % filter_size, initializer=self.initializer
            ):
                # Convolution Layer
                if hparams.use_entity and hparams.use_context:
                    filter_shape = [filter_size, dim * 3, 1, num_filters]
                elif hparams.use_entity:
                    filter_shape = [filter_size, dim * 2, 1, num_filters]
                else:
                    filter_shape = [filter_size, dim, 1, num_filters]
                W = tf.compat.v1.get_variable(
                    name="W" + "_filter_size_" + str(filter_size),
                    shape=filter_shape,
                    dtype=tf.float32,
                    initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                        scale=1.0,
                        mode="fan_avg",
                        distribution=("uniform" if False else "truncated_normal"),
                    ),
                )
                b = tf.compat.v1.get_variable(
                    name="b" + "_filter_size_" + str(filter_size),
                    shape=[num_filters],
                    dtype=tf.float32,
                )
                if W not in self.layer_params:
                    self.layer_params.append(W)
                if b not in self.layer_params:
                    self.layer_params.append(b)
                conv = tf.nn.conv2d(
                    input=concat_expanded,
                    filters=W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv",
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool2d(
                    h,
                    ksize=[1, hparams.doc_size - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool",
                )
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        # self.num_filters_total is the kims cnn output dimension
        self.num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, axis=-1)
        h_pool_flat = tf.reshape(h_pool, [-1, self.num_filters_total])
        return h_pool_flat

    def infer_embedding(self, sess, feed_dict):
        """Infer document embedding in feed_dict with current model.

        Args:
            sess (object): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: News embedding in a batch.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.news_field_embed_final_batch], feed_dict=feed_dict)

    def run_get_embedding(self, infile_name, outfile_name):
        """infer document embedding with current model.

        Args:
            infile_name (str): Input file name, format is [Newsid] [w1,w2,w3...] [e1,e2,e3...]
            outfile_name (str): Output file name, format is [Newsid] [embedding]

        Returns:
            object: An instance of self.
        """
        load_sess = self.sess
        with tf.io.gfile.GFile(outfile_name, "w") as wt:
            for (
                batch_data_input,
                newsid_list,
                data_size,
            ) in self.iterator.load_infer_data_from_file(infile_name):
                news_embedding = self.infer_embedding(load_sess, batch_data_input)[0]
                for i in range(data_size):
                    wt.write(
                        newsid_list[i]
                        + " "
                        + ",".join(
                            [
                                str(embedding_value)
                                for embedding_value in news_embedding[i]
                            ]
                        )
                        + "\n"
                    )
        return self
