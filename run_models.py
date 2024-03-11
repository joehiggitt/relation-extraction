import argparse
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


LABELS = ["no_relation", "per:identity", "per:title", "per:employee_of",
    "org:top_members/employees", "org:alternate_names",
    "org:country_of_branch", "org:city_of_branch", "org:members", "per:age",
    "per:origin", "per:spouse", "org:member_of",
    "org:stateorprovince_of_branch", "per:date_of_death",
    "per:countries_of_residence", "per:children", "per:cause_of_death",
    "per:stateorprovinces_of_residence", "per:cities_of_residence",
    "per:city_of_death", "per:parents", "per:siblings",
    "org:political/religious_affiliation", "per:charges", "org:website",
    "per:schools_attended", "org:founded_by", "org:shareholders",
    "per:religion", "per:other_family", "per:city_of_birth", "org:founded",
    "per:stateorprovince_of_death", "per:date_of_birth",
    "org:number_of_employees/members", "per:stateorprovince_of_birth",
    "per:country_of_death", "per:country_of_birth", "org:dissolved"
]

TEXT_EMEDDING_SIZE = 300
POS_EMEDDING_SIZE = 100
LSTM_UNITS = 128

VOCAB_SIZE = 20000
MAX_LEN = 100
NUM_LABELS = 39


def classify_binary_out(output: np.ndarray) -> np.ndarray:
    """
    Returns the prediction of a binary classifier output.

    Parameters
    ----------
    `output`: `np.ndarray`
        The outputted classifications.

    Returns
    -------
    `np.ndarray`
        The predicted classification labels (based on if the prediction is
        greater than 0.5 or not).
    """
    return tf.nest.flatten(tf.cast(tf.math.greater_equal(tf.cast(output, dtype=tf.float16), tf.constant(0.5, dtype=tf.float16)), dtype=tf.uint8))

def classify_multiclass_out(output: np.ndarray) -> np.ndarray:
    """
    Returns the prediction of a multi-class classifier output.

    Parameters
    ----------
    `output`: `np.ndarray`
        The outputted classifications.

    Returns
    -------
    `np.ndarray`
        The predicted classification labels (based on the label with the
        greatest probability).
    """
    return tf.argmax(output, axis=1)

def valid_int(x: int) -> bool:
    return (x >= 0) and (x < MAX_LEN)

def parse_pos(string: str) -> tuple[bool, None | tuple[int, int]]:
    try:
        int_string = int(string)
    except ValueError:

        split_string = re.split(" |-", string)
        try:
            split_string = [int(item) for item in split_string]
        except ValueError:
            return False, None
        
        if valid_int(split_string[0]) and valid_int(split_string[1]):
            print((split_string[0], split_string[1]))
            return True, (split_string[0], split_string[1])
        
        return False, None
    
    if valid_int(int_string):
        return True, (int_string, int_string)
    
    return False, None
        
def create_pos_array(length, a, b):
    """
    Creates a relative position indexing array of `length`, with 0s
    placed between `a` and `b` (inclusive). For indexes `i < a`,
    `i = i - a`. For indexes `i > b`, `i = i - b`.

    Parameters
    ----------
    `length`: `int`
        The length of the output array.
    `a`: `int`
        The start index of the 0 region.
    `b`: `int`
        The end index of the 0 region.

    Returns
    -------
    `np.ndarray` of shape `(length,)`
        A NumPy array containing the relative positions.

    Raises
    ------
    `ValueError`
        If `a > b`.
    """
    if (a > b):
        raise ValueError("`a` must be less than or equal to `b`.")
    pos = np.zeros(length, dtype=np.int8)
    pos[b + 1:] = np.arange(1, length - b)
    pos[:a] = np.arange(a, 0, -1) * -1

    return pos


def run_inference(model: tf.keras.Model):
    run_loop = True
    while run_loop == True:
        print("Enter a sentence to parse. (\Q to quit)")
        sentence = input("> ")

        if (sentence == r"\Q"):
            run_loop = False
            continue

        sentence_vec = tf.reshape(tf.convert_to_tensor(sentence, dtype=tf.string), (1,))

        is_valid = False
        while is_valid == False:
            print("Enter the position of the subject (either a single number or an inclusive range in the form 'a-b').")
            valid, subj = parse_pos(input("> "))
            subj_pos = create_pos_array(MAX_LEN, *subj).reshape(1, MAX_LEN)
            is_valid = valid
            
        is_valid = False
        while is_valid == False:
            print("Enter the position of the object (either a single number or a range in the form 'a-b').")
            valid, obj = parse_pos(input("> "))
            obj_pos = create_pos_array(MAX_LEN, *obj).reshape(1, MAX_LEN)
            is_valid = valid

        prediction_probs = model.predict([sentence_vec, subj_pos, obj_pos], verbose=0)
        predictions = classify_multiclass_out(prediction_probs)
        print(f"\nPredicted class: {LABELS[predictions[0]]}\n\n")
    


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser(prog="run_models.py", description="Queries RelExLSTM model.")
    parser.add_argument("-m", "--model", required=True, help="Keras model filepath.")
    args = parser.parse_args()

    if (args.model is None):
        print("No keras model filepath provided.")
        exit()

    model = load_model(args.model)

    run_inference(model)
