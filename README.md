# Relation Extraction with Bidirectional LSTM Model
An exploration of models capable of extracting relations between entities in text.

This README provides instructions on how to install and use the relation extraction model.

## Version
This tool is currently at version 1.0.

## Requirements
This tool uses Python 3.11.7, with the following packages:
- Tensorflow 2.15.0
- NumPy 1.26.4
- Pandas 2.2.1
- Matplotlib 3.8.3
- Sklearn 1.4.1

The required packages are listed in `requirements.txt`.

## Installation
Install the latest versions of Python and PIP, instructions for which can be found at the following websites:

- Python installation: https://www.python.org/downloads/
- PIP installation: https://pip.pypa.io/en/stable/installation/

Clone the git repository onto the server or computer you want to run the detector on.
```
$ git clone <your git clone address>
$ cd relation-extraction
```
Once cloned, install the Python requirements with PIP.
```
$ pip install -r requirements.txt
```

## Project Structure
This project repository contains the following:

    relation-extraction/
    ├── checkpoints/        - contains model training checkpoints
    ├── keras_models/       - contains .keras model files
    ├── models/
    │   ├── model1.py       - code for initialising first model
    │   └── model2.py       - code for initialising second model
    ├── .gitignore
    ├── LSTM.ipynb          - Jupyter notebook detailing training/testing of model
    ├── README.md
    ├── requirements.txt    - list of project requirements
    └── run_models.py       - code for running CLI

## Dataset
This code was trained on the Re-TACRED Dataset, which is a modified version of the TACRED Dataset. These modifications address the shortcomings of the original dataset

The original TACRED Dataset is available for download from the LDC: https://catalog.ldc.upenn.edu/LDC2018T24. It is free for members, or $25 for non-members.

The modifications can be made to the dataset using the code hosted on https://github.com/gstoica27/Re-TACRED. Follow the instructions in their README.md file to run the modifications code.

## Running the Code
### CLI Usage
To run the program using `run_models.py`, use the following command:

    $ python run_models.py -m "model_filepath"

The command line interface has the following options:

    -m, --model "model_filepath"
        Filepath to the .keras model file. Required.
    -h, --help
        Shows a help message and exits.


### Using with Other Code

#### Creating New Model
`model2.py` can be used to instanstiate an LSTM. The following code shows how a model can created and used.

```
from model2 import create_RelExLSTM, create_vectoriser, 

MAX_LEN = 100

vectoriser, vocab = create_vectoriser(
    corpus=,
    vocab_size=,
    max_len=MAX_LEN,
    use_vocab=,
)

model = create_RelExLSTM(
    text_embed_size=,
    pos_embed_size=,
    lstm_units=,
    vocab_size=,
    num_labels=,
    max_len=MAX_LEN,
)

model.build([(MAX_LEN,), (MAX_LEN,), (MAX_LEN,)])
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
model.summary()

full_model = create_RelExLSTM_with_vectoriser(model, vectoriser, max_len=MAX_LEN)
```

#### Loading Pre-Trained Model
A pre-trained model can also be loaded into code as follows:

```
from tensorflow.keras.models import load_model

model_filepath = ...
full_model = load_model(model_filepath)
```


## Jupyter Notebook
The Jupyter Notebook - `LSTM.ipynb` - was used during development to train and test various models.

The notebook can be re-run to recreate the training process with alternate parameters. Instructions are contained within the notebook.

There is an inference mode at the bottom of the notebook, where the model can be queried. This uses the CLI in `run_models.py`.
