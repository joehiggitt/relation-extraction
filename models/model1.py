import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import InputLayer, TextVectorization, Embedding, Bidirectional, LSTM, Dense


class RelExLSTM(Model):
    def __init__(self, embedding_size, lstm_units, vocab_size, num_labels, max_len, vocab, use_vocab=False):
        super(RelExLSTM, self).__init__()

        self.input_layer = InputLayer(name="Input", input_shape=(1,), dtype=tf.string)

        if use_vocab:
            self.vectoriser = TextVectorization(name="TextVectorisation", vocabulary=vocab, output_sequence_length=max_len)
        else:
            self.vectoriser = TextVectorization(name="TextVectorisation", max_tokens=vocab_size, output_sequence_length=max_len)
            self.vectoriser.adapt(vocab)
            
        self.embedding = Embedding(vocab_size, embedding_size, input_length=max_len, name="Embedding")
        self.lstm = Bidirectional(LSTM(lstm_units, dropout=0.7, recurrent_dropout=0.7), name="BidirectionalLSTM")
        self.dense = Dense(num_labels, activation='softmax', name="FullyConnected")

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.vectoriser(x)
        x = self.embedding(x)
        x = self.lstm(x)
        y = self.dense(x)
        return y