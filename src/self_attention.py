from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Dropout, Add, Conv1D
from keras.initializers import Ones, Zeros


class PositionEmbedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # must be a even number
        self.mode = mode
        super(PositionEmbedding, self).__init__(**kwargs)

    def call(self, x):
        if self.size is None or self.mode == 'sum':
            self.size = int(x.shape[-1])
        # batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange does not support variable length
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class MultiHeadAttention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def mask(inputs, seq_len, mode='mul'):
        if seq_len is None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x, Q_len=None, V_len=None):
        # if only Q_seq, K_seq & V_seq are given(Q_len & V_len are not given), then no mask
        # if Q_seq, K_seq, V_seq, Q_len & V_len are all given, then mask the redundant
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        else:
            raise RuntimeError('not support parameter number!')

        # linear transformation for  Q, K & V
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))

        # calculate inner produc, then mask & softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)

        # output & mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.output_dim


class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:], initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:], initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)

    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    def compute_output_shape(self, input_shape):
        return input_shape


class PositionWiseFeedForward(Layer):
    def __init__(self, filters, dropout=0.1, **kwargs):
        self.conv1 = Conv1D(filters, 1, activation='tanh', padding='same')
        self.conv2 = Conv1D(filters, 1, activation='tanh', padding='same')
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    def call(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)


class SelfAttentionEncoder(Layer):
    def __init__(self, nb_head, size_per_head, dropout_rate=0.5, use_norm=True, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        self.dropout_rate = dropout_rate
        self.use_norm = use_norm
        super(SelfAttentionEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.self_attention = MultiHeadAttention(self.nb_head, self.size_per_head)
        self.dropout = Dropout(self.dropout_rate)
        self.add = Add()
        self.norm = LayerNormalization()
        self.feed_forward = PositionWiseFeedForward(self.output_dim)

    def call(self, x):
        outputs = self.self_attention([x, x, x])
        outputs = self.dropout(outputs)
        outputs = self.add([outputs, x])
        if self.use_norm:
            outputs = self.norm(outputs)
        outputs = self.feed_forward(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return self.self_attention.compute_output_shape(input_shape)
