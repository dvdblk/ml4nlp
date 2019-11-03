# Imports

import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import nltk
import itertools
import time, os
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from argparse import Namespace


class Vocabulary:
    def __init__(self, add_unk=True):
        self._token_to_ids = {}
        self._ids_to_token = {}

        if add_unk:
            self.unk_index = self.add_token("<UNK>")
        else:
            self.unk_index = -1


    def vocabulary_set(self):
        """this function returns a list of unique tokens"""
        return(list(set(self.tokens)))

    def add_token(self, token):
        """Update mapping dicts based on the token.

        Args:
            token (str): the item to add into the Vocabulary
        Returns:
            index (int): the integer corresponding to the token
        """
        if token in self._token_to_ids:
            index = self._token_to_ids[token]
        else:
            index = len(self._token_to_ids)
            self._token_to_ids[token] = index
            self._ids_to_token[index] = token
        return index

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_ids.get(token, self.unk_index)
        else:
            return self._token_to_ids[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._ids_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._ids_to_token[index]

    def __len__(self):
        return len(self._token_to_ids)

class Vectorizer(object):
    def __init__(self, vocabulary):
        self.vocab = vocabulary

    @classmethod
    def from_dataframe(cls, cbow_df):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            cbow_df (pandas.DataFrame): the target dataset
        Returns:
            an instance of the Vectorizer
        """
        vocabulary = Vocabulary()
        for index, row in cbow_df.iterrows():
            # add each context word (token) to the vocabulary
            for token in row.context:
                vocabulary.add_token(token)

            # add the target word as well
            vocabulary.add_token(row.target)

        return cls(vocabulary)

    def vectorize(self, context_words):
        context_ids = [self.vocab.lookup_token(w) for w in context_words]
        return torch.tensor(context_ids, dtype=torch.long)

class ShakespeareDataset(Dataset):
    def __init__(self, cbow_df):
        """
        Args:
            cbow_df (pandas.DataFrame): the dataset as a dataframe
        """
        # 98/1/1% split
        self.train_df, self.val_df, self.test_df = \
          np.split(cbow_df, [int(.98*len(cbow_df)), int(.99*len(cbow_df))])

        self._lookup_dict = {'train': self.train_df,
                             'val': self.val_df,
                             'test': self.test_df}

        self.set_split()
        self._vectorizer = Vectorizer.from_dataframe(self.train_df)

    @classmethod
    def load_and_create_dataset(cls, filepath, context_size, frac=1.0):
        """Load and preprocess the dataset

        Args:
            filepath (str): location of the dataset
            context_size (int): size of the context before/after the target word
            frac (float, optional): fraction of the data to use (default 1.0)
        Returns:
            an instance of ShakespeareDataset
        """
        # load the file
        lines = ShakespeareDataset._load_file(filepath)
        # consider the fraction param and throw away the rest
        lines = lines[:int(len(lines)*frac)]

        # Preprocess
        tokens = ShakespeareDataset._preprocess_and_split_lines(lines)

        # Create DataFrame
        dataframe_data = ShakespeareDataset._create_context_data(
            tokens,
            context_size
        )
        cbow_df = pd.DataFrame(dataframe_data, columns=['context', 'target'])

        # Create an instance
        return cls(cbow_df)

    @staticmethod
    def _load_file(filepath):
        """Load the dataset file into lines"""
        with open(filepath) as file:
            lines = file.readlines()
            file.close()
            return lines

    @staticmethod
    def _preprocess_and_split_lines(lines):
        """

        Args:
            lines (list): a list of lines of the dataset
        Returns:
            a list of tokens
        """

        # Regex
        lines = lines[134:164924] #these numbers are only valid for the full corpus
        text = ''.join(lines)
        text = re.sub(r'\d+', '', text)
        text = re.sub('SCENE \S', '', text)
        text = re.sub('(\[_).*(_\])', '', text)
        text = re.sub(r'[\\[#$%*+—/<=>?{}|~@]+_', '', text)
        text = text.lower()

        # Tokenize
        tokens = nltk.tokenize.word_tokenize(text)
        #tokens = text.split()

        return tokens

    @staticmethod
    def _create_context_data(tokens, context_size):
        data = []
        for i in range(context_size, len(tokens) - context_size):
            # Context before w_i
            context_before_w = tokens[i - context_size: i]

            # Context after w_i
            context_after_w = tokens[i + 1: i + context_size + 1]

            # Put them together
            context_window = context_before_w + context_after_w

            # Target = w_i
            target = tokens[i]

            # Append in the correct format
            data.append([context_window, target])
        return data

    def get_vectorizer(self):
        """ returns the vectorizer """
        return self._vectorizer

    def set_split(self, split="train"):
        """ selects the splits in the dataset using a column in the dataframe """
        self._target_df = self._lookup_dict[split]

    def __len__(self):
        return len(self._target_df)

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features (x_data) and label (y_target)
        """
        row = self._target_df.iloc[index]

        context_vector = self._vectorizer.vectorize(row.context)
        target_index = self._vectorizer.vocab.lookup_token(row.target)

        return {'x_data': context_vector,
                'y_target': target_index}

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size, nr_hidden_neurons=128):
        super(CBOW, self).__init__()
        self._context_window_size = context_size * 2

        # Embedding/input layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Hidden layer
        self.linear1 = nn.Linear(embedding_dim, nr_hidden_neurons)

        # Output layer
        self.linear2 = nn.Linear(nr_hidden_neurons, vocab_size)


    def forward(self, inputs):
        # shape = (WINDOW_SIZE, EMBEDDING_DIM) -> (EMBEDDING_DIM)
        embeds = self.embeddings(inputs).sum(dim=1)

        # finally compute the hidden layer weighted sum (a.k.a. output before using the activation function)
        # ... and don't forget to divide by the number of input vectors
        h =  self.linear1(embeds) / self._context_window_size

        # output of the hidden layer
        out =  F.relu(h)

        # output
        # also note that we don't compute softmax here because Cross Entropy is used as a loss function
        out = F.relu(self.linear2(out))
        return out

class TrainState:

    EPOCH_INDEX = "epoch_index"
    TRAIN_LOSS = "train_loss"
    VAL_LOSS = "val_loss"
    MODEL_FP = "model_filepath"

    @staticmethod
    def make_train_state(filepath):
        """Return a new train state (dict)"""
        return {
            TrainState.EPOCH_INDEX: 0,
            TrainState.TRAIN_LOSS: [],
            TrainState.VAL_LOSS: [],
            TrainState.MODEL_FP: filepath
        }

    @staticmethod
    def save_model(train_state, model):
        """Handle the training state updates.

        model (nn.Module): model to save
        """
        # Save the model at least once
        if train_state.get(TrainState.EPOCH_INDEX) == 0:
            torch.save(model.state_dict(), train_state.get(TrainState.MODEL_FP))

        # Save model if performance improves
        else:
            loss_prev, loss_cur = train_state.get(TrainState.VAL_LOSS)[-2:]

            # compare current loss with the previous one
            if loss_cur <= loss_prev:
                # save if needed
                torch.save(
                    model.state_dict(),
                    train_state.get(TrainState.MODEL_FP)
                )

class TrainingRoutine:

    EMBEDDING_DIM = "embedding_dim"
    CONTEXT_SIZE = "context_size"
    NR_HIDDEN = "nr_hidden_neurons"
    LEARNING_RATE = "learning_rate"
    DATA_FRAC = "data_frac"
    NR_EPOCHS = "nr_epochs"
    BATCH_SIZE = "batch_size"

    @staticmethod
    def create_training_args(
        embedding_dim, context_size, nr_hidden_neurons,
        learning_rate, data_frac, nr_epochs, batch_size):
        return {
            TrainingRoutine.EMBEDDING_DIM: embedding_dim,
            TrainingRoutine.CONTEXT_SIZE: context_size,
            TrainingRoutine.NR_HIDDEN: nr_hidden_neurons,
            TrainingRoutine.LEARNING_RATE: learning_rate,
            TrainingRoutine.DATA_FRAC: float(data_frac),
            TrainingRoutine.NR_EPOCHS: nr_epochs,
            TrainingRoutine.BATCH_SIZE: batch_size
        }

    def __init__(self, shakespeare_csv_filepath, embedding_dim, context_size,
                nr_hidden_neurons, data_frac, device, learning_rate,
                filedir, filepath, dataset=None):
        # Create dataset if needed, othw reuse
        if dataset is None:
            new_dataset = ShakespeareDataset.load_and_create_dataset(
                shakespeare_csv_filepath,
                context_size,
                data_frac
            )
            self.dataset = new_dataset
        else:
            self.dataset = dataset

        # Classifier
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        vocab_len = len(self.dataset.get_vectorizer().vocab)
        classifier = CBOW(
            vocab_len,
            embedding_dim,
            context_size,
            nr_hidden_neurons
        )
        self.classifier = classifier.to(device)
        self.optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

        filename = str(context_size) + "_" + str(nr_hidden_neurons) + "_" + str(learning_rate) + "_" + filepath
        self.train_state = TrainState.make_train_state(filedir + filename)


    def _train_step(self, batch_size):
        # Iterate over training dataset
        # setup: batch generator, set loss to 0, set train mode on
        self.dataset.set_split('train')
        batch_generator = generate_batches(self.dataset,
                                          batch_size=batch_size,
                                          device=self.device)
        running_loss = 0.0
        # make sure our weights / embeddings get updated
        self.classifier.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            # step 1. zero the gradients
            self.optimizer.zero_grad()

            # step 2. compute the output
            y_pred = self.classifier(batch_dict['x_data'])

            # step 3. compute the loss
            loss = self.loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

            # step 4. use loss to produce gradients
            loss.backward()

            # step 5. use optimizer to take gradient step
            self.optimizer.step()

        return running_loss


    def _val_step(self, batch_size):
        # Iterate over val dataset
        # setup: batch generator, set loss to 0; set eval mode on
        self.dataset.set_split('val')
        batch_generator = generate_batches(self.dataset,
                                          batch_size=batch_size,
                                          device=self.device)
        running_loss = 0.0
        self.classifier.eval()

        for batch_index, batch_dict in enumerate(batch_generator):

            # compute the output
            y_pred =  self.classifier(batch_dict['x_data'])

            # compute the loss
            loss = self.loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

        return running_loss

    def train(self, num_epochs, batch_size):
      for epoch_index in tqdm(range(num_epochs)):
          self.train_state[TrainState.EPOCH_INDEX] = epoch_index

          train_loss = self._train_step(batch_size)
          self.train_state[TrainState.TRAIN_LOSS].append(train_loss)

          val_loss = self._val_step(batch_size)
          self.train_state[TrainState.VAL_LOSS].append(val_loss)

          TrainState.save_model(self.train_state, self.classifier)

def generate_batches(dataset, batch_size, shuffle=True,
                     drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
      ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)
        yield out_data_dict


def start_training_routine(args, training_args, model_state_dir, dataset=None):
    training_routine = TrainingRoutine(
        args.shakespeare_csv_filepath,
        training_args.get(TrainingRoutine.EMBEDDING_DIM),
        training_args.get(TrainingRoutine.CONTEXT_SIZE),
        training_args.get(TrainingRoutine.NR_HIDDEN),
        training_args.get(TrainingRoutine.DATA_FRAC),
        args.device,
        training_args.get(TrainingRoutine.LEARNING_RATE),
        model_state_dir,
        args.model_state_file,
        dataset=dataset
    )

    training_routine.train(
      training_args.get(TrainingRoutine.NR_EPOCHS),
      training_args.get(TrainingRoutine.BATCH_SIZE)
    )

    return training_routine

def setup(args):
    # Check CUDA
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("Using CUDA: {}".format(args.cuda))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

def get_args():
    return Namespace(
        # Data and Path information
        shakespeare_csv_filepath="shakespeare-corpus.txt",
        model_state_file="shakespeare_model.pth",
        model_state_dir="models/",
        # Runtime options
        seed=1337,
        cuda=True
    )

# Sequential grid search
def grid_search(args, grid_search_params):
    classifiers = []

    values = [lists for _, lists in grid_search_params.items()]
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_gridsearch_dir = args.model_state_dir + "gridsearch/"
    model_dir = model_gridsearch_dir + timestr + "_" \
        + str(training_args.num_epochs) + "/"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    def dataset_key(context_size, frac):
        return str(context_size) + "_" + str(frac)

    datasets = dict()

    # Cartesian product
    for lr, nr_hidden in itertools.product(*values):
        # try to reuse the dataset if it exists
        maybe_dataset = datasets.get(dataset_key(context_size, frac))
        training_routine = start_training_routine(maybe_dataset)

        dataset = training_routine.dataset
        if maybe_dataset != dataset:
            # Save the dataset if it's a new one
            datasets[dataset_key(dataset)] = dataset

        classifiers.append((training_routine.classifier, lr))

    return classifiers

def train_best_models(args):
    # CBOW with context size = 2
    cbow2_training_args = TrainingRoutine.create_training_args(
        embedding_dim=50,
        context_size=2,
        nr_hidden_neurons=128,
        learning_rate=0.01,
        data_frac=0.001,
        nr_epochs=200,
        batch_size=32
    )

    cbow2_training_routine = start_training_routine(
        args, cbow2_training_args, args.model_state_dir
    )

    # CBOW with context size = 5
    cbow5_training_args = cbow2_training_args.copy()
    cbow5_training_args[TrainingRoutine.CONTEXT_SIZE] = 5

    cbow5_training_routine = start_training_routine(
        args, cbow5_training_args, args.model_state_dir
    )

TRAIN = True
GRID_SEARCH = False
BEST_MODEL_FP = ""

if __name__ == "__main__":
    args = get_args()
    setup(args)

    if TRAIN:
        if GRID_SEARCH:
            #grid_search_params = {
            #      TrainingRoutine.LEARNING_RATE: [0.001],
            #      TrainingRoutine.NR_EPOCHS: [50, 100]
            #}
            #grid_search(args, grid_search_params)
            pass
        else:
            train_best_models(args)
    else:
        # eval / show closest words for BEST_MODEL
        pass
