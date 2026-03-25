import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Concatenate, Dropout, Add,
    MultiHeadAttention, Activation, Lambda, GlobalMaxPooling1D,
    Multiply, Dot, Flatten, LayerNormalization
)
from tensorflow.keras.models import Model


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='linear'),
        tf.keras.layers.Dense(d_model, activation='linear')
    ])


class CoAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dff, dropout_rate=0.1,
                 epsilon=1e-6, input_dim=None, **kwargs):
        super(CoAttentionBlock, self).__init__(**kwargs)
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=key_dim
        )
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        self.layer_norm = LayerNormalization(epsilon=epsilon)
        self.ffn = point_wise_feed_forward_network(input_dim, dff)
        self.flatten = Flatten()

    def call(self, inputs, training=None):
        text_expanded, other_expanded = inputs

        co_attention = self.multi_head_attention(
            text_expanded, other_expanded, other_expanded
        )
        co_attention = self.dropout(co_attention, training=training)
        co_attention = self.add([co_attention, text_expanded])
        co_attention = self.layer_norm(co_attention)

        co_attention_ffn = self.ffn(co_attention)
        co_attention_ffn = self.dropout(co_attention_ffn, training=training)
        co_attention_ffn = self.add([co_attention_ffn, co_attention])
        co_attention_ffn = self.layer_norm(co_attention_ffn)

        return co_attention_ffn


class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dff, dropout_rate=0.1,
                 epsilon=1e-6, input_dim=None, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.multi_head_attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=key_dim
        )
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        self.layer_norm = LayerNormalization(epsilon=epsilon)
        self.ffn = point_wise_feed_forward_network(input_dim, dff)
        self.flatten = Flatten()

    def call(self, inputs, training=None):
        self_attention = self.multi_head_attention(inputs, inputs, inputs)
        self_attention = self_attention = self.dropout(self_attention, training=training)
        self_attention = self.add([self_attention, inputs])
        self_attention = self.layer_norm(self_attention)

        self_attention_ffn = self.ffn(self_attention)
        self_attention_ffn = self.dropout(self_attention_ffn, training=training)
        self_attention_ffn = self.add([self_attention_ffn, self_attention])
        self_attention_ffn = self.layer_norm(self_attention_ffn)

        return self_attention_ffn


def build_surefar_model(
    config,
    vocab_size,
    embedding_matrix,
    input_dim,
    textrank_bool=True,
    bart_bool=True,
    output_dim=300,
    num_heads=4,
    dff=1024,
    fusion_version='gmu',
    gmu_activation='sigmoid',
    mlp_depth=2,
    mlp_hidden_dim=128,
    dropout_rate=0.2
):
    """
    Reflects the paper/original implementation structure:
    1) TextRank (Extractive): user/item -> embedding -> pooling -> concat -> projection -> self-attn
    2) BART (Abstractive): user/item -> concat -> projection -> self-attn
    3) Extractive vs Abstractive fusion (GMU by default)
    4) MLP -> Rating prediction
    """

    model_params = config.get('model_params', {})
    output_dim = model_params.get('embedding_dim', output_dim)
    mlp_hidden_dim = model_params.get('mlp_hidden_dim', mlp_hidden_dim)
    mlp_depth = model_params.get('mlp_depth', mlp_depth)
    dropout_rate = model_params.get('dropout_rate', dropout_rate)

    key_dim = output_dim // num_heads

    self_attn_block = SelfAttentionBlock(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout_rate=dropout_rate,
        dff=dff,
        input_dim=output_dim,
        name="self_attention_block"
    )

    co_attn_block = CoAttentionBlock(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout_rate=dropout_rate,
        dff=dff,
        input_dim=output_dim,
        name="co_attention_block"
    )

    input_list = []

    # --------------------------------------------------
    # 1. TextRank / Extractive branch
    # --------------------------------------------------
    if textrank_bool:
        user_ext_in = Input(shape=(input_dim,), dtype='int32', name='user_ext_seq')
        item_ext_in = Input(shape=(input_dim,), dtype='int32', name='item_ext_seq')
        input_list += [user_ext_in, item_ext_in]

        glove_embedding = Embedding(
            input_dim=vocab_size,
            output_dim=output_dim,
            input_length=input_dim,
            weights=[embedding_matrix],
            trainable=False,
            name='glove_embedding'
        )

        user_ext_emb = glove_embedding(user_ext_in)
        item_ext_emb = glove_embedding(item_ext_in)

        user_ext_pool = GlobalMaxPooling1D(name='user_ext_pool')(user_ext_emb)
        item_ext_pool = GlobalMaxPooling1D(name='item_ext_pool')(item_ext_emb)

        ext_concat = Concatenate(name='ext_concat')([user_ext_pool, item_ext_pool])
        ext_proj = Dense(output_dim, activation='linear', name='ext_projection')(ext_concat)

        ext_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1), name='ext_expand')(ext_proj)
        ext_attn = self_attn_block(ext_expanded)
        ext_input = Lambda(lambda x: tf.squeeze(x, axis=1), name='ext_squeeze')(ext_attn)
    else:
        ext_input = None

    # --------------------------------------------------
    # 2. BART / Abstractive branch
    # --------------------------------------------------
    if bart_bool:
        user_abs_in = Input(shape=(768,), dtype='float32', name='user_abs_vec')
        item_abs_in = Input(shape=(768,), dtype='float32', name='item_abs_vec')
        input_list += [user_abs_in, item_abs_in]

        abs_concat = Concatenate(name='abs_concat')([user_abs_in, item_abs_in])
        abs_proj = Dense(output_dim, activation='linear', name='abs_projection')(abs_concat)

        abs_expanded = Lambda(lambda x: tf.expand_dims(x, axis=1), name='abs_expand')(abs_proj)
        abs_attn = self_attn_block(abs_expanded)
        abs_input = Lambda(lambda x: tf.squeeze(x, axis=1), name='abs_squeeze')(abs_attn)
    else:
        abs_input = None

    # --------------------------------------------------
    # 3. Fusion: Extractive vs Abstractive
    # --------------------------------------------------
    if textrank_bool and bart_bool:
        fusion_version = fusion_version.lower()

        if fusion_version == 'gmu':
            concat_inputs = Concatenate(name='concat_inputs')([ext_input, abs_input])
            gate_sum = Dense(output_dim, activation=gmu_activation, name='gmu_gate')(concat_inputs)

            tanh_ext = Activation('tanh', name='tanh_ext')(ext_input)
            tanh_abs = Activation('tanh', name='tanh_abs')(abs_input)

            gate_inv = Lambda(lambda x: 1.0 - x, name='gate_inverse')(gate_sum)

            gmu_output = Add(name='gmu_output')([
                Multiply(name='gmu_ext')([gate_sum, tanh_ext]),
                Multiply(name='gmu_abs')([gate_inv, tanh_abs])
            ])
            combined = gmu_output

        elif fusion_version == 'concat':
            combined = Concatenate(name='fusion_concat')([ext_input, abs_input])

        elif fusion_version.startswith('inner'):
            combined = Dot(axes=-1, name='fusion_inner_product')([ext_input, abs_input])

        elif fusion_version.startswith('element'):
            combined = Multiply(name='fusion_elementwise')([ext_input, abs_input])

        elif fusion_version.startswith('at'):
            ext_exp = Lambda(lambda x: tf.expand_dims(x, axis=1), name='fusion_ext_expand')(ext_input)
            abs_exp = Lambda(lambda x: tf.expand_dims(x, axis=1), name='fusion_abs_expand')(abs_input)

            co_attn_ext = co_attn_block((ext_exp, abs_exp))
            co_attn_abs = co_attn_block((abs_exp, ext_exp))

            combined = Concatenate(name='co_attn_output')([co_attn_ext, co_attn_abs])
            combined = Lambda(lambda x: tf.squeeze(x, axis=1), name='co_attn_squeeze')(combined)

        else:
            raise ValueError(f"Invalid fusion_version: {fusion_version}")

    elif textrank_bool:
        combined = ext_input
    elif bart_bool:
        combined = abs_input
    else:
        raise ValueError("At least one of textrank_bool or bart_bool must be True.")

    # --------------------------------------------------
    # 4. MLP -> Rating prediction
    # --------------------------------------------------
    mlp_hidden_units = [max(mlp_hidden_dim // (2 ** i), 8) for i in range(mlp_depth)]

    x = combined
    for i, units in enumerate(mlp_hidden_units):
        x = Dense(units, activation='linear', name=f'mlp_dense_{i+1}')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate, name=f'dropout_{i+1}')(x)

    rating_pred = Dense(1, activation='relu', name='rating_output')(x)

    model = Model(
        inputs=input_list,
        outputs=rating_pred,
        name=f'{fusion_version}_rating_model'
    )

    return model