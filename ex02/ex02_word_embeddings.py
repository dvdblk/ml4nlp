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
        lines = lines[134:164924]
        text = ''.join(lines)
        text = re.sub(r'\d+', '', text)
        text = re.sub('SCENE \S', '', text)
        text = re.sub('(\[_).*(_\])', '', text)
        text = re.sub(r'[\\[#$%*+—/<=>?{}|~@]+_', '', text)
        text = text.lower()

        # Tokenize
        tokens = nltk.tokenize.word_tokenize(text)

        return tokens

    @staticmethod
    def _create_context_data(tokens, context_size):
        """Create the data which will be fed into the classifier"""
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
    """Continuous Bag of Words Classifier"""

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

class FSUtil:
    """File system utilities"""

    SEPARATOR = "_"

    @staticmethod
    def create_dir_if_needed(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    @staticmethod
    def get_model_filename(context_size, embedding_dim, nr_hidden,
        learning_rate, nr_epochs, generic_filename="model.pth"):
        strings = list(map(str, [
            context_size, embedding_dim, nr_hidden, learning_rate, nr_epochs
        ]))
        return FSUtil.SEPARATOR.join(
            strings + [generic_filename]
        )

    @staticmethod
    def create_gridsearch_subdir(models_dir, gs_dir, nr_iterations):
        ite_str = str(nr_iterations) + "iterations" + FSUtil.SEPARATOR
        gridsearch_dir = os.path.join(models_dir, gs_dir)
        final_dir = os.path.join(gridsearch_dir, time.strftime("%Y%m%d-%H%M%S"))
        FSUtil.create_dir_if_needed(final_dir)
        return final_dir


class TrainState:
    """Utility methods and constants for the trainstate dictionary"""

    EPOCH_INDEX = "epoch_index"
    TRAIN_LOSS = "train_loss"
    VAL_LOSS = "val_loss"
    MODEL_DIR = "model_directory"

    @staticmethod
    def make_train_state(model_dir):
        """Return a new train state (dict)"""
        return {
            TrainState.EPOCH_INDEX: 0,
            TrainState.TRAIN_LOSS: [],
            TrainState.VAL_LOSS: [],
            TrainState.MODEL_DIR: model_dir
        }

    @staticmethod
    def save_model(train_state, training_args, model):
        """Handle the training state updates.

        model (nn.Module): model to save
        """
        filename = FSUtil.get_model_filename(
            training_args.get(TrainingRoutine.CONTEXT_SIZE),
            training_args.get(TrainingRoutine.EMBEDDING_DIM),
            training_args.get(TrainingRoutine.NR_HIDDEN),
            training_args.get(TrainingRoutine.LEARNING_RATE),
            training_args.get(TrainingRoutine.NR_EPOCHS)
        )
        filepath = os.path.join(
            train_state.get(TrainState.MODEL_DIR), filename
        )
        # Save the model at least once
        if train_state.get(TrainState.EPOCH_INDEX) == 0:
            torch.save(model.state_dict(), filepath)

        # Save model if performance improves
        else:
            loss_prev, loss_cur = train_state.get(TrainState.VAL_LOSS)[-2:]

            # compare current loss with the previous one
            if loss_cur <= loss_prev:
                # save if needed
                torch.save(
                    model.state_dict(),
                    filepath
                )

class TrainingRoutine:
    """Encapsulates the training of a CBOW classifier"""

    EMBEDDING_DIM = "embedding_dim"
    CONTEXT_SIZE = "context_size"
    NR_HIDDEN = "nr_hidden_neurons"
    LEARNING_RATE = "learning_rate"
    DATA_FRAC = "data_frac"
    NR_EPOCHS = "nr_epochs"
    BATCH_SIZE = "batch_size"

    def __init__(self, shakespeare_csv_filepath, training_args, device,
                filedir, dataset=None):
        self.training_args = training_args
        embedding_dim = training_args.get(TrainingRoutine.EMBEDDING_DIM)
        context_size = training_args.get(TrainingRoutine.CONTEXT_SIZE)
        nr_hidden_neurons = training_args.get(TrainingRoutine.NR_HIDDEN)
        data_frac = training_args.get(TrainingRoutine.DATA_FRAC)
        learning_rate = training_args.get(TrainingRoutine.LEARNING_RATE)

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

        self.device = device

        # Model
        self.loss_func = nn.CrossEntropyLoss()
        vocab_len = len(self.dataset.get_vectorizer().vocab)
        model = CBOW(
            vocab_len,
            embedding_dim,
            context_size,
            nr_hidden_neurons
        )
        self.model = model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_state = TrainState.make_train_state(filedir)

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

    @staticmethod
    def create_default_training_args():
        """Create default training arguments"""
        return TrainingRoutine.create_training_args(
            embedding_dim=50,
            context_size=2,
            nr_hidden_neurons=128,
            learning_rate=0.01,
            data_frac=0.001,
            nr_epochs=10,
            batch_size=32
        )

    @staticmethod
    def start_training_routine(
        args, training_args, model_state_dir, dataset=None):
        """Create and train a training routine

        Returns:
            training_routine (TrainingRoutine): the trained routine
        """
        training_routine = TrainingRoutine(
            args.shakespeare_csv_filepath,
            training_args,
            args.device,
            model_state_dir,
            dataset=dataset
        )

        training_routine.train(training_args)

        return training_routine


    def _train_step(self, batch_size):
        """Do a training iteration over the batch"""
        # Iterate over training dataset
        # setup: batch generator, set loss to 0, set train mode on
        self.dataset.set_split('train')
        batch_generator = generate_batches(self.dataset,
                                          batch_size=batch_size,
                                          device=self.device)
        running_loss = 0.0
        # make sure our weights / embeddings get updated
        self.model.train()

        for batch_index, batch_dict in enumerate(batch_generator):
            # step 1. zero the gradients
            self.optimizer.zero_grad()

            # step 2. compute the output
            y_pred = self.model(batch_dict['x_data'])

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
        """Do a validation iteration over the batch"""
        # Iterate over val dataset
        # setup: batch generator, set loss to 0; set eval mode on
        self.dataset.set_split('val')
        batch_generator = generate_batches(self.dataset,
                                          batch_size=batch_size,
                                          device=self.device)
        running_loss = 0.0
        self.model.eval()

        for batch_index, batch_dict in enumerate(batch_generator):

            # compute the output
            y_pred =  self.model(batch_dict['x_data'])

            # compute the loss
            loss = self.loss_func(y_pred, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index + 1)

        return running_loss

    def train(self, training_args):
        """Do the proper training steps"""
        nr_epochs = training_args.get(TrainingRoutine.NR_EPOCHS)
        batch_size = training_args.get(TrainingRoutine.BATCH_SIZE)

        for epoch_index in tqdm(range(nr_epochs)):
            self.train_state[TrainState.EPOCH_INDEX] = epoch_index

            train_loss = self._train_step(batch_size)
            self.train_state[TrainState.TRAIN_LOSS].append(train_loss)

            val_loss = self._val_step(batch_size)
            self.train_state[TrainState.VAL_LOSS].append(val_loss)

            TrainState.save_model(
                self.train_state, training_args, self.model
            )

class SavedDatasets(dict):

    @staticmethod
    def dataset_key(context_size, frac):
        """Return the key for given arguments

        Args:
            context_size (int): the size of the context in the dataset
            frac (float): % of dataset that is used
        """
        return str(context_size) + "_" + str(frac)



class GridSearch:
    """A gridsearch routine that encapsulates necessary methods"""

    def __init__(self):
        self.routines = []
        self.saved_datasets = SavedDatasets()

    @staticmethod
    def _get_param_combinations(grid_search_params):
        """Return a list of dictionaries with each parameter combination"""
        param_pools = [
            [ (param, c) for c in choices ]
            for param, choices in grid_search_params.items()
        ]

        return list(map(dict, itertools.product(*param_pools)))


    def start(self, args, grid_search_params):
        """Start a sequential grid GridSearch

        Returns:
            routines (list): a list of executed training routines
        """

        param_combinations = GridSearch._get_param_combinations(
            grid_search_params
        )

        # Directory to save the classifiers into
        classifier_dir = FSUtil.create_gridsearch_subdir(
            args.model_state_dir,
            args.gridsearch_dir,
            len(param_combinations)
        )

        for param_dict in param_combinations:
            # Create default training parameters
            training_args = TrainingRoutine.create_default_training_args()

            # Update accordingly to the current grid search param values
            for param, value in param_dict.items():
                training_args[param] = value

            context_size = training_args.get(TrainingRoutine.CONTEXT_SIZE)
            learning_rate = training_args.get(TrainingRoutine.LEARNING_RATE)
            data_frac = training_args.get(TrainingRoutine.DATA_FRAC)

            # try to reuse the dataset if it exists
            maybe_dataset = self.saved_datasets.get(
                SavedDatasets.dataset_key(context_size, data_frac)
            )

            # start the training routine
            routine = TrainingRoutine.start_training_routine(
                args, training_args, classifier_dir, maybe_dataset
            )

            # Save the dataset if it's a new one so we can reuse it later
            self._save_dataset_if_needed(
                maybe_dataset,
                routine.dataset,
                data_frac
            )

            # Append the trained classifier into the result
            self.routines.append(routine)
        return self.routines

    def _save_dataset_if_needed(self, maybe_dataset, dataset, data_frac):
        """Save the dataset if it isn't present in the saved_datasets dict"""
        if maybe_dataset != dataset:
            key = SavedDatasets.dataset_key(dataset, data_frac)
            self.saved_datasets[key] = dataset


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

def setup(args):
    """Runtime/Env setup"""
    # create the models directory for saving
    FSUtil.create_dir_if_needed(args.model_state_dir)

    # NLTK
    nltk.download('punkt')

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
        gridsearch_dir="gridsearch/",
        # Runtime options
        seed=1337,
        cuda=True
    )


def train_best_models(args):
    """Train both classifiers with the best hyperparams"""
    # CBOW with context size = 2
    cbow2_training_args = TrainingRoutine.create_default_training_args()
    cbow5_training_args[TrainingRoutine.CONTEXT_SIZE] = 2

    cbow2_training_routine = TrainingRoutine.start_training_routine(
        args, cbow2_training_args, args.model_state_dir
    )

    # CBOW with context size = 5
    cbow5_training_args = cbow2_training_args.copy()
    cbow5_training_args[TrainingRoutine.CONTEXT_SIZE] = 5

    cbow5_training_routine = TrainingRoutine.start_training_routine(
        args, cbow5_training_args, args.model_state_dir
    )

    return [cbow2_training_routine,
            cbow5_training_routine]

class CBOWEvaluator:

    def __init__(self, args):
        self._args = args
        self.dataset = None


    def _get_model_from_file(self, filepath):
        """Loads the model from a file.

        Returns:
            (model, ctx_size, nr_hidden, lr, nr_epochs) \
            (CBOW, int, int, float, int): a tuple
        """
        # get the number of neurons from filename
        filename = os.path.basename(filepath)
        str_ctx, str_embedding_dim, str_hidden, str_lr, str_epoch, *rest = filename.split("_")
        context_size = int(str_ctx)

        if self.dataset is None:
            print("Creating the dataset, this might take a while (1-2min)...")
            self.dataset = ShakespeareDataset.load_and_create_dataset(
                self._args.shakespeare_csv_filepath,
                context_size
            )

        # init the model
        vocab_len = len(self.dataset.get_vectorizer().vocab)
        model = CBOW(
            vocab_len, int(str_embedding_dim),
            context_size, int(str_hidden)
        )
        # load the weights / embeddings
        model.load_state_dict(torch.load(filepath, map_location=torch.device('cpu')))
        # set to eval mode
        model.eval()

        return (
            model, context_size,
            int(str_hidden), float(str_lr), int(str_epoch)
        )

    def get_closest_word_generic(self, model, dataset, word,
        similarity_measure, descending, topn):
        word_distance = []
        emb = model.embeddings
        vocab = dataset.get_vectorizer().vocab

        i = vocab.lookup_token(word)
        lookup_tensor_i = torch.tensor([i], dtype=torch.long).to(
            self._args.device
        )
        v_i = emb(lookup_tensor_i)
        for j in range(len(vocab)):
            if j != i:
                lookup_tensor_j = torch.tensor([j], dtype=torch.long).to(
                    self._args.device
                )
                v_j = emb(lookup_tensor_j)
                word_distance.append(
                    (vocab.lookup_index(j), float(similarity_measure(v_i, v_j)))
                )
        word_distance.sort(key=lambda x: x[1])
        if descending:
            return word_distance[::-1][:topn]
        else:
            return word_distance[:topn]

    def evaluate_model(self, model, word, learning_rate,
                        context_size, nr_epochs, topn=5, dataset=None):
        """Pretty print the model evaluation

        Args:
            model (CBOW): the model
            word (string): the word to evaluate
            learning_rate (float): the learning_rate of the model
            context_size (int): the context size
            batch_size (int): the size of training batches
            topn (int, optional): the number of closest words to show
            dataset (ShakespeareDataset): the dataset to use
        """
        # Get variables
        get_closest_word_pwd = self.get_closest_word_generic(
            model, dataset, word, nn.PairwiseDistance(), False, topn
        )
        get_closest_word_cs = self.get_closest_word_generic(
            model, dataset, word, nn.CosineSimilarity(), True, topn
        )

        # Print
        print("=" * 50)
        print("Model (Learning Rate: " + str(learning_rate) + ", Epochs: " \
            + str(nr_epochs) + "): " + str(model) + "\n")
        print("===Pairwise Distance (lower better)===")
        self._pretty_print(get_closest_word_pwd)
        print("===Cosine Similarity (higher better)===")
        self._pretty_print(get_closest_word_cs)


    def evaluate_model_from_file(self, filepath, word, topn=5):
        model, context_size, _, lr, epochs = self._get_model_from_file(filepath)
        self.evaluate_model(
            model, word, lr, context_size, epochs, topn, self.dataset
        )

    def evaluate_routine(self, training_routine, word, topn=5):
        self.evaluate_model(
            training_routine.model,
            word,
            training_routine.training_args.get(TrainingRoutine.LEARNING_RATE),
            training_routine.training_args.get(TrainingRoutine.CONTEXT_SIZE),
            training_routine.training_args.get(TrainingRoutine.NR_EPOCHS),
            topn,
            training_routine.dataset
        )


    def evaluate_gridsearch_dir(self, gridsearch_dir, word, topn=5):
        """Evaluate the last gridsearch training directory"""
        grid_search_directories = os.listdir(gridsearch_dir)
        grid_search_directories.sort()
        last_grid_search_dir = grid_search_directories[-1]
        target_dir = os.path.join(gridsearch_dir, last_grid_search_dir)

        models_loaded = []

        for filename in os.listdir(target_dir):
            if filename.endswith(".pth"):
                self.evaluate_model_from_file(filename, word, topn)

    def _pretty_print(self, results):
        """
        Pretty print embedding results.
        """
        for item in results:
            print ("...[%.2f] - %s"%(item[1], item[0]))

# //
# Dear Reviewer, feel free to play around with these 4 constants...
# //
TRAIN = True
GRID_SEARCH = True
EVAL = True
EVAL_GRIDSEARCH_DIR = False

# Entrypoint
if __name__ == "__main__":
    # Setup
    args = get_args()
    setup(args)

    routines = []

    if TRAIN:
        if GRID_SEARCH:
            # Start grid search
            grid_search_params = {
                TrainingRoutine.LEARNING_RATE: [0.001],
                TrainingRoutine.NR_EPOCHS: [3, 50],
                TrainingRoutine.DATA_FRAC: [0.001]
            }
            grid_search = GridSearch()
            routines = grid_search.start(args, grid_search_params)
        else:
            # Train the best model
            routines = train_best_models(args)


    evaluator = CBOWEvaluator(args)


    word = "happiness"
    while word != "q":

        print("=" * 50)

        if EVAL:
            if not routines:
                # evaluate the given file
                # eval / show closest words
                model_fp = "models/2_50_128_0.01_200_shakespeare_model.pth"
                print("Evaluating " + word + " on " + model_fp)
                evaluator.evaluate_model_from_file(model_fp, word)
            else:
                print("Evaluating " + word + " on trained routines.")
                # if routines is not an empty list, evaluate them
                for routine in routines:
                    evaluator.evaluate_routine(routine, word)

        if EVAL_GRIDSEARCH_DIR:
            print("Evaluating " + word + " on last gridsearch directory.")
            evaluator.evaluate_gridsearch_dir(
                os.path.join(args.model_state_dir, args.gridsearch_dir),
                word
            )
        word = input("Enter word to evaluate (q to quit): ")
