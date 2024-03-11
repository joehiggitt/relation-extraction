import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Bidirectional, Dense, Embedding,
    InputLayer, LSTM, TextVectorization)


class RelExLSTM(Model):
    def __init__(self, embedding_size, lstm_units, num_labels, vocab_size, vocab,
        max_len=100, use_vocab=False):
        """
        Constructs a Bidirectional LSTM neural network, desgined for
        identifying relations in sentences
        
        Parameters
        ----------
        `embedding_size`: `int`
            Dimensionality of the vectors returned from the embedding layer.
        `lstm_units`: `int`
            Number of hidden units in LSTM layer.
        `num_labels`: `int`
            Number of classes for relation classification.
        `vocab_size`: `int`
            Size of the vocabulary being used.
        `vocab`: `list` of `str`, `str`, `tf.data.Dataset`, `np.ndarray`
            A vocabulary for use in `TextVectorization` (which can be an
            array of strings or a filepath to a text file), or a corpus
            to create a new vocabulary from (which can be a `Dataset` or
            a `NDArray`).
        `max_len`: `int`
            Maximum number of tokens allowed in a sentence.
        `use_vocab`: `bool`
            Whether `vocab` is a pre-created vocabulary for use in 
            `TextVectorization` (`True`), or a corpus to create a new
            vocabulary from (`False`).

        Returns
        -------
        `RelExLSTM`
            A Bidirectional LSTM model.
        """
        super(RelExLSTM, self).__init__()

        # Input Layer
        self.input_layer = InputLayer(name="Input", input_shape=(1,),
            dtype=tf.string)

        # Text Vectorisation Layer
        if use_vocab:
            self.vectoriser = TextVectorization(name="TextVectorisation",
                vocabulary=vocab, output_sequence_length=max_len)
        else:
            self.vectoriser = TextVectorization(name="TextVectorisation",
                max_tokens=vocab_size, output_sequence_length=max_len)
            self.vectoriser.adapt(vocab)
            
        # Text Embedding Layer
        self.embedding = Embedding(vocab_size, embedding_size,
            input_length=max_len, name="Embedding")
        
        # Bidirection LSTM Layers
        self.lstm = Bidirectional(LSTM(lstm_units, dropout=0.7,
            recurrent_dropout=0.7), name="BidirectionalLSTM")
        self.dense = Dense(num_labels, activation='softmax',
            name="FullyConnected")

    def call(self, inputs):
        x = self.input_layer(inputs)
        x = self.vectoriser(x)
        x = self.embedding(x)
        x = self.lstm(x)
        y = self.dense(x)
        return y