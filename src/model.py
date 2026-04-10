import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Concatenate, Dropout, Dense, Add, 
    MultiHeadAttention, Lambda, GlobalMaxPooling1D, Multiply, Activation
)
from tensorflow.keras.models import Model

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='linear'),
        tf.keras.layers.Dense(d_model, activation='linear')
    ])

class CoAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dff, dropout_rate=0.1, epsilon=1e-6, input_dim=None, **kwargs):
        super(CoAttentionBlock, self).__init__(**kwargs)
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=key_dim)
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.ffn = point_wise_feed_forward_network(input_dim, dff)

    def call(self, inputs):
        text_expanded, image_expanded = inputs
        co_attention = self.multi_head_attention(text_expanded, image_expanded, image_expanded)
        co_attention = self.dropout(co_attention)
        co_attention = self.add([co_attention, text_expanded])
        co_attention = self.layer_norm(co_attention)

        co_attention_ffn = self.ffn(co_attention)
        co_attention_ffn = self.dropout(co_attention_ffn)
        co_attention_ffn = self.add([co_attention_ffn, co_attention])
        co_attention_ffn = self.layer_norm(co_attention_ffn)
        return co_attention_ffn

class SelfAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dff, dropout_rate=0.1, epsilon=1e-6, input_dim=None, **kwargs):
        super(SelfAttentionBlock, self).__init__(**kwargs)
        self.multi_head_attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=key_dim)
        self.dropout = Dropout(dropout_rate)
        self.add = Add()
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=epsilon)
        self.ffn = point_wise_feed_forward_network(input_dim, dff)

    def call(self, inputs):
        information_expanded = inputs
        self_attention = self.multi_head_attention(information_expanded, information_expanded, information_expanded)
        self_attention = self.dropout(self_attention)
        self_attention = self.add([self_attention, information_expanded])
        self_attention = self.layer_norm(self_attention)

        self_attention_ffn = self.ffn(self_attention)
        self_attention_ffn = self.dropout(self_attention_ffn)
        self_attention_ffn = self.add([self_attention_ffn, self_attention])
        self_attention_ffn = self.layer_norm(self_attention_ffn)
        return self_attention_ffn

def build_proposed_model(params, textrank_user_embedding_matrix=None, textrank_item_embedding_matrix=None):
    """
    params 딕셔너리를 입력받아 제안된 추천 모델 아키텍처를 빌드합니다.
    """
    inputs = []
    features_to_fuse = []
    
    output_dim = params['embedding_dim']
    num_heads = params.get('num_heads', 4)
    dropout_rate = params.get('dropout_rate', 0.2)

    # 1. TextRank Module
    if params.get('textrank_bool', True):
        user_tr_in = Input(shape=(params['input_dim'],), dtype='int32', name='user_textrank_input')
        item_tr_in = Input(shape=(params['input_dim'],), dtype='int32', name='item_textrank_input')
        inputs.extend([user_tr_in, item_tr_in])

        user_tr_emb = Embedding(input_dim=len(textrank_user_embedding_matrix), output_dim=output_dim, 
                                weights=[textrank_user_embedding_matrix], trainable=False)(user_tr_in)
        item_tr_emb = Embedding(input_dim=len(textrank_item_embedding_matrix), output_dim=output_dim, 
                                weights=[textrank_item_embedding_matrix], trainable=False)(item_tr_in)

        user_tr_pool = GlobalMaxPooling1D()(user_tr_emb)
        item_tr_pool = GlobalMaxPooling1D()(item_tr_emb)
        
        tr_concat = Concatenate()([user_tr_pool, item_tr_pool])
        tr_proj = Dense(output_dim, activation='relu')(tr_concat)
        features_to_fuse.append(tr_proj)

    # 2. BART Module
    if params.get('bart_bool', True):
        user_bart_in = Input(shape=(768,), dtype='float32', name='user_bart_input')
        item_bart_in = Input(shape=(768,), dtype='float32', name='item_bart_input')
        inputs.extend([user_bart_in, item_bart_in])
        
        bart_concat = Concatenate()([user_bart_in, item_bart_in])
        bart_proj = Dense(output_dim, activation='relu')(bart_concat)
        features_to_fuse.append(bart_proj)

    # 3. Attention Module
    if len(features_to_fuse) == 2:
        if params.get('use_attention', True):
            key_dim = output_dim // num_heads
            self_attn = SelfAttentionBlock(num_heads=num_heads, key_dim=key_dim, dff=params.get('dff', 1024), 
                                           input_dim=output_dim, dropout_rate=dropout_rate)
            
            tr_expanded = tf.expand_dims(features_to_fuse[0], axis=1)
            bart_expanded = tf.expand_dims(features_to_fuse[1], axis=1)
            
            feat1 = tf.squeeze(self_attn(tr_expanded), axis=1)
            feat2 = tf.squeeze(self_attn(bart_expanded), axis=1)
        else:
            feat1, feat2 = features_to_fuse[0], features_to_fuse[1]

        # 4. Fusion Module
        fusion_version = params.get('fusion_version', 'gmu')
        if fusion_version == 'gmu':
            concat_for_gate = Concatenate()([feat1, feat2])
            gmu_gate = Dense(output_dim, activation='sigmoid', name='gmu_gate')(concat_for_gate)
            
            feat1_act = Activation('tanh')(feat1)
            feat2_act = Activation('tanh')(feat2)
            
            one_minus_gate = Lambda(lambda x: 1.0 - x)(gmu_gate)
            
            gmu_feat1 = Multiply()([gmu_gate, feat1_act])
            gmu_feat2 = Multiply()([one_minus_gate, feat2_act])
            combined = Add()([gmu_feat1, gmu_feat2])
        else:
            combined = Concatenate()([feat1, feat2])
    else:
        combined = features_to_fuse[0]

    # 5. MLP Rating Predictor
    mlp_hidden_units = [max(1, params['mlp_hidden_dim'] // (2 ** i)) for i in range(params['mlp_depth'])]
    x = combined
    for i, units in enumerate(mlp_hidden_units):
        x = Dense(units, activation='relu', name=f'mlp_dense_{i+1}')(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    rating_pred = Dense(1, activation='linear', name='rating_output')(x)
    
    model_name = f"{params.get('fusion_version', 'single')}_rating_model"
    
    return Model(inputs=inputs, outputs=rating_pred, name=model_name)