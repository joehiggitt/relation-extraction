import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Bidirectional, Concatenate, Dense,
    Embedding, LSTM, TextVectorization)

def create_RelExLSTM(text_embed_size: int, pos_embed_size: int,
    lstm_units: int, max_len: int, num_labels: int, vocab_size: int):
    """
    Constructs a Bidirectional LSTM neural network, desgined for
    identifying relations in sentences
    
    Parameters
    ----------
    `text_embed_size`: `int`
        Dimensionality of the vectors returned from the text embedding
        layer.
    `pos_embed_size`: `int`
        Dimensionality of the vectors returned from the subject/object
        position embedding layers.
    `lstm_units`: `int`
        Number of hidden units in LSTM layer.
    `max_len`: `int`
        Maximum number of tokens allowed in a sentence.
    `num_labels`: `int`
        Number of classes for relation classification.
    `vocab_size`: `int`
        Size of the vocabulary being used.

    Returns
    -------
    `tf.keras.Model`
        A Bidirectional LSTM model.
    """
    # Determines activation based on whether classification is binary or
    # multi-class
    if num_labels == 1:
        activation = "sigmoid"
    else:
        activation = "softmax"

    # Input Layers
    text_input = Input(shape=(max_len,), name="TextInput")
    pos1_input = Input(shape=(max_len,), name="Position1Input")
    pos2_input = Input(shape=(max_len,), name="Position2Input")

    # Embedding Layers
    text_embedding = Embedding(vocab_size, text_embed_size, input_length=
        max_len, name="TextEmbedding")(text_input)
    pos1_embedding = Embedding(max_len, pos_embed_size, input_length=max_len,
        name="Pos1Embedding")(pos1_input)
    pos2_embedding = Embedding(max_len, pos_embed_size, input_length=max_len,
        name="Pos2Embedding")(pos2_input)

    # Concatenation Layer
    concat_out = Concatenate(name="Concatenate")([text_embedding, pos1_embedding,
        pos2_embedding])

    # Bidirectional LSTM Layers
    lstm_out = Bidirectional(LSTM(lstm_units, dropout=0.7), name="BidirectionalLSTM"
        )(concat_out)
    out = Dense(num_labels, activation=activation, name="FullyConnected")(
        lstm_out)
    
    return Model(
        inputs=[text_input, pos1_input, pos2_input], 
        outputs=out
    )

def create_vectoriser(corpus, vocab_size, max_len, use_vocab):
    """
        Constructs a `tf.keras.layers.TextVectorization` layer for
        preprocessing text for the LSTM model.
        
        Parameters
        ----------
        `corpus`: `list` of `str`, `str`, `tf.data.Dataset`, `np.ndarray`
            A vocabulary for use in `TextVectorization` (which can be an
            array of strings or a filepath to a text file), or a corpus
            to create a new vocabulary from (which can be a `Dataset` or
            a `NDArray`).
        `vocab_size`: `int`
            Size of the vocabulary being used.
        `max_len`: `int`
            Maximum number of tokens allowed in a sentence.
        `use_vocab`: `bool`
            Whether `vocab` is a pre-created vocabulary for use in 
            `tf.keras.layers.TextVectorization` (`True`), or a corpus to
            create a new vocabulary from (`False`).

        Returns
        -------
        `tf.keras.layers.TextVectorization`
            A `tf.keras.layers.TextVectorization` layer.
        `np.ndarray`
            The vocabulary used by the
            `tf.keras.layers.TextVectorization` layer.
    """
    if use_vocab:
        # Defines vectoriser
        vectoriser = TextVectorization(max_tokens=vocab_size, 
            output_sequence_length=max_len, vocabulary=corpus,
            name="TextVectorisation")
    
    else:
        # Defines vectoriser
        vectoriser = TextVectorization(max_tokens=vocab_size, 
            output_sequence_length=max_len, name="TextVectorisation")
        
        # Trains vectoriser on corpus
        vectoriser.adapt(corpus)
    
    return vectoriser, vectoriser.get_vocabulary()

def create_RelExLSTM_with_vectoriser(lstm: Model, vectoriser: 
    TextVectorization, max_len: int):
    """
    Constructs a Bidirectional LSTM neural network with a preprocessing
    `tf.keras.layers.TextVectorization` layer, desgined for identifying
    relations in sentences.
    
    Parameters
    ----------
    `lstm`: `tf.keras.Model`
        RelExLSTM model.
    `vectoriser`: `tf.keras.layers.TextVectorization`
        Text vectorisation layer.
    `max_len`: `int`
        Maximum number of tokens allowed in a sentence.

    Returns
    -------
    `tf.keras.Model`
        A Bidirectional LSTM model.
    """
    # Input Layers
    text_input = Input(shape=(1,), dtype=tf.string, name="TextInput")
    pos1_input = Input(shape=(max_len,), name="Position1Input")
    pos2_input = Input(shape=(max_len,), name="Position2Input")

    # Combined models
    text_input_vec = vectoriser(text_input)
    out = lstm([text_input_vec, pos1_input, pos2_input])

    return Model(
        inputs=[text_input, pos1_input, pos2_input],
        outputs=out
    )