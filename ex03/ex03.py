# Imports

import json
import numpy as np
import pandas as pd
import re
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from tqdm import tqdm
import os
import time
import datetime
import html

# constants
# Column names
COL_ID = 'ID'
COL_TWEET = 'Tweet'
COL_LABEL = 'Label'

# minimum number of instances that we require to be present in the training set
# for a given language to be included in fitting of the model
MIN_NR_OCCURENCES = 1000

# unknown class name
CLASS_UNK = '<UNK>'

class Vocabulary:
    def __init__(self, add_unk=True):
        self._token_to_ids = {}
        self._ids_to_token = {}

        # Add the unknown token and index
        if add_unk:
            self.unk_index = self.add_token(CLASS_UNK)
        else:
            self.unk_index = -1

    def vocabulary_set(self):
        """this function returns a list of unique tokens"""
        return self._ids_to_token.values()

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


class TweetsVocabulary(Vocabulary):

    def add_tweet(self, tweet):
        """ Convenience function for adding the entire tweet to the vocabulary.

        Args:
            tweet (str): the tweet to add into the TweetsVocabulary
        Returns:
            token_indices (list): list of integers corresponding to the token
                                    indexes
        """
        token_indices = []

        # Each character of a tweet becomes a token
        for token in list(tweet):
          index = self.add_token(token)
          token_indices.append(index)

        return token_indices

    def lookup_tweet(self, tweet):
        """ Convenience function for looking up the indexes of a tweet
        in the vocabulary.

        Args:
            tweet (str): the tweet to lookup
        Returns:
            indices (torch.tensor): int64 tensor of the tweet token indices
        """
        indices = []

        for token in list(tweet):
            index = self.lookup_token(token)
            indices.append(index)

        return torch.tensor(indices, dtype=torch.int64)

class Vectorizer:
    """ The Vectorizer which coordinates the Vocabularies and creates
    an input vector.
    """

    def __init__(self, token_vocab, label_vocab,
        max_tweet_length):
        """
        Args:
            surname_vocab (Vocabulary): maps characters to integers
            nationality_vocab (Vocabulary): maps nationalities to integers
            max_surname_length (int): the length of the longest surname
        """
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self._nr_of_tokens = len(self.token_vocab)
        self._max_tweet_length = max_tweet_length

    def vectorize(self, tweet):
        """
        Args:
            tweet (list): the tweet
        Returns:
            result (torch.Tensor): a tensor of one-hot vectors,
                                padded to _max_tweet_length
        """
        result = torch.zeros(self._max_tweet_length, self._nr_of_tokens)
        one_hot_matrix = nn.functional.one_hot(
            self.token_vocab.lookup_tweet(tweet),
            num_classes=self._nr_of_tokens
        )
        result[:len(one_hot_matrix), :] = one_hot_matrix
        return result

    @classmethod
    def from_dataframe(cls, tweets_df, max_twt_length):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            surname_df (pandas.DataFrame): the surnames dataset
        Returns:
            an instance of the SurnameVectorizer
        """
        tokens_vocab = TweetsVocabulary()
        labels_vocab = Vocabulary()

        for index, row in tweets_df.iterrows():
          tokens_vocab.add_tweet(row.Tweet)
          labels_vocab.add_token(row.Label)

        return cls(tokens_vocab, labels_vocab, max_twt_length)


class TweetsDataset(Dataset):

    INPUT_X="x_data"
    TARGET_Y="y_target"

    def __init__(self, dataframes, max_twt_length):
        """
        Args : I don't really know yet, this init is not ready as of yet
        """
        self.train_df = dataframes[0]
        self.dev_df = dataframes[1]
        self.test_df = dataframes[2]

        self._lookup_dict = {'train' : self.train_df,
                             'val': self.dev_df,
                             'test': self.test_df}

        self._vectorizer = Vectorizer.from_dataframe(
            self.train_df, max_twt_length
        )
        self.set_split()

        # FIXME:
        #class weight --> needed?
        #self.class_weights = 1.0 / torch.tensor(frequencies, dtype=torch.float32)


    @classmethod
    def load_and_create_dataset(cls, tweets_fp, train_fp, test_fp,
        max_tweet_length, data_frac=1, train_dev_frac=0.9):
        """
        Args: filepath
        """
        tweets = TweetsDataset._get_tweets(tweets_fp)
        train_labels = TweetsDataset._get_train_labels(train_fp)
        test_labels = TweetsDataset._get_test_labels(test_fp)
        data = TweetsDataset._create_sets(
            tweets, train_labels, test_labels, data_frac, train_dev_frac,
        )
        return cls(data, max_tweet_length)

    @staticmethod
    def _create_sets(
        tweets, train_dev_labels, test_labels, data_frac, train_dev_frac):
        """Return a tuple of dataframes comprising three main data sets"""

        # to allow for merge, need the same type
        tweets[COL_ID] = tweets[COL_ID].astype(np.int64)

        # Merge by ID
        train_dev_data = pd.merge(tweets, train_dev_labels, on=COL_ID)
        test_set = pd.merge(tweets, test_labels, on=COL_ID)
        take_part_of_df = lambda df: np.split(df, [int(data_frac*len(df))])
        train_dev_data, _ = take_part_of_df(train_dev_data)
        test_set, _ = take_part_of_df(test_set)

        # take (train_dev_frac * 100) % of the traindevdata
        train_set = train_dev_data.sample(frac=train_dev_frac, random_state=0)
        # take % that remain
        dev_set = train_dev_data.drop(train_set.index)

        # drop the ID columns, not needed anymore
        train = train_set.drop(COL_ID, axis=1)
        dev = dev_set.drop(COL_ID, axis=1)
        test = test_set.drop(COL_ID, axis=1)

        return train, dev, test

    @staticmethod
    def _get_train_labels(filepath):
        """Return a dataframe of train_dev labels"""
        train_dev_labels = pd.read_csv(filepath, sep='\t', header=None,
            names=[COL_LABEL, COL_ID]
        )
        # remove whitespace from labels (e.g. "ar  " should be equal to "ar")
        train_dev_labels.Label = train_dev_labels.Label.str.strip()

        # remove 'und' languages
        und_indices = train_dev_labels.index[train_dev_labels.Label == 'und']
        train_dev_labels.drop(und_indices, inplace=True)

        # deal with class imbalance in the train set
        lang_occurence = train_dev_labels.groupby(COL_LABEL).size()
        balanced_languages = lang_occurence.where(
            lang_occurence >= MIN_NR_OCCURENCES
        ).dropna().index.values
        balanced_labels = train_dev_labels.Label.isin(balanced_languages)

        # Option 1 - replace rows that are labelled with an imbalanced language
        # ~ is element-wise logical not
        #train_dev_labels.loc[~balanced_labels, COL_LABEL] = CLASS_UNK

        # Option 2 - keep the rows that are labelled with a balanced language
        train_dev_labels = train_dev_labels[balanced_labels]
        return train_dev_labels

    @staticmethod
    def _get_test_labels(filepath):
        """Return a dataframe of test labels"""
        return pd.read_csv(filepath, sep='\t', header=None,
            names=[COL_LABEL, COL_ID]
        )

    @staticmethod
    def _get_tweets(filepath):
        """Return a dataframe of tweets"""
        tweets = []
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)

        with open(filepath, 'r', encoding="utf-8") as tweets_fh:  # Tweets file handle
            for line in tweets_fh:   # put each line in a list of lines
                j_content = json.loads(line)
                #preprocessing steps first!
                text = j_content[1]
                text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
                text = re.sub('@\S*', '', text)
                text = html.unescape(text)
                if text:
                    # if the tweet text is not empty
                    j_content[1] = text
                    tweets.append(j_content)

        # make a dataframe out of it
        tweets = pd.DataFrame(tweets, columns=[COL_ID, COL_TWEET])
        return tweets

    def get_vectorizer(self):
        """returns the vectorizer"""
        return self._vectorizer

    def set_split(self, split = "train"):
        """selects the splits in the dataset"""
        self._target_df = self._lookup_dict[split]

    def __len__(self):
        return len(self._target_df)

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point's features / input vector
            (x_data) and label (y_target)
        """
        vectorizer = self.get_vectorizer()
        row = self._target_df.iloc[index]
        one_hotted_tweet = vectorizer.vectorize(row.Tweet)
        tweet_lang_target_index = vectorizer.label_vocab.lookup_token(row.Label)
        return {TweetsDataset.INPUT_X: one_hotted_tweet,
                TweetsDataset.TARGET_Y: tweet_lang_target_index}

    def generate_batches(self, batch_size, device, shuffle=True,
                         drop_last=True):
        """
        A generator function which wraps the PyTorch DataLoader. It will
          ensure each tensor is on the write device location.
        """
        dataloader = DataLoader(dataset=self, batch_size=batch_size,
                                shuffle=shuffle, drop_last=drop_last)

        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = tensor.to(device)
            yield (
                out_data_dict[TweetsDataset.INPUT_X],
                out_data_dict[TweetsDataset.TARGET_Y]
            )


class TweetClassifier(nn.Module):

    def __init__(self, input_length, nr_filters, kernel_length, one_hot_length,
                nr_hidden, nr_languages):
        super(TweetClassifier, self).__init__()
        # Convolution layer
        self.conv1 = torch.nn.Conv1d(one_hot_length, nr_filters, kernel_length)
        # Pooling
        self.pool = torch.nn.MaxPool1d(input_length-1)
        # Fully Connected layers
        self.fc1 = torch.nn.Linear(nr_filters, nr_hidden)
        self.fc2 = torch.nn.Linear(nr_hidden, nr_languages)

    def forward(self, x):
        #print("x (shape: {}): {}".format(x.shape, x))
        out = x.permute(0, 2, 1)
        #print("x permuted (shape: {}): {}".format(out.shape, out))
        out = F.relu(self.conv1(out))
        #print("conv1 (shape: {}): {}".format(out.shape, out))
        out = self.pool(out)
        #print("pool (shape: {}): {}".format(out.shape, out))
        out = torch.flatten(out, 1, 2)
        #print("pool flattened (shape: {}): {}".format(out.shape, out))
        out = F.relu(self.fc1(out))
        #print("fc1 (shape: {}): {}".format(out.shape, out))
        out = F.relu(self.fc2(out))
        #print("fc2/output (shape: {}): {}".format(out.shape, out))
        return out


class FSUtil:
    """File system utilities"""

    SEPARATOR = "_"

    @staticmethod
    def get_model_filename(nr_hidden,
        learning_rate, nr_epochs, generic_filename="model.pth"):
        strings = list(map(str, [
            nr_hidden, learning_rate, nr_epochs
        ]))
        return FSUtil.SEPARATOR.join(
            strings + [generic_filename]
        )

class TrainingRoutine:
    """Encapsulates the training of a classifier"""
    MAX_TWEET_LENGTH = 256

    def __init__(self, args, nr_filters, kernel_length, nr_hidden, nr_epochs,
                batch_size, learning_rate, data_frac, train_dev_frac):
        self.nr_hidden_neurons = nr_hidden
        self.nr_filters = nr_filters,
        self.kernel_length = kernel_length
        self.nr_epochs = nr_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.dataset = TweetsDataset.load_and_create_dataset(
            args.tweets_fp,
            args.train_dev_fp,
            args.test_fp,
            TrainingRoutine.MAX_TWEET_LENGTH,
            data_frac,
            train_dev_frac
        )

        self.device = args.device

        # Model
        self.loss_func = nn.CrossEntropyLoss()
        vectorizer = self.dataset.get_vectorizer()
        vocab_len = len(vectorizer.token_vocab)
        nr_languages = len(vectorizer.label_vocab)
        model = TweetClassifier(
            input_length=TrainingRoutine.MAX_TWEET_LENGTH,
            nr_filters=nr_filters,
            kernel_length=kernel_length,
            one_hot_length=vocab_len,
            nr_hidden=nr_hidden,
            nr_languages=nr_languages
        )
        self.model = model.to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def start_training_routine(self, args):
        """Create and train a training routine

        Returns:
            training_routine (TrainingRoutine): the trained routine
        """
        start = time.time()
        print_time = lambda str: print("[{0:%Y-%m-%d %H:%M:%S}] {1}".format(
            datetime.datetime.now(), str
        ))

        # Start the training
        print(" Training ".center(80, "="))
        self.train(args)
        end = time.time()
        print_time("\nTraining finished in {0:.{1}f} seconds.\n".format(
            end - start, 4
        ))


    def _train_step(self):
        """Do a training iteration over the batch"""
        training_bar = tqdm(desc='Batch',
                          total=len(self.dataset) // self.batch_size,
                          position=1,
                          leave=False)
        # setup: batch generator, set train mode on
        self.dataset.set_split('train')
        batch_generator = self.dataset.generate_batches(device=self.device,
                                                    batch_size=self.batch_size)

        # make sure our weights / embeddings get updated
        self.model.train()

        for batch_index, (batch_X, batch_y) in enumerate(batch_generator):
            # step 1. zero the gradients
            self.optimizer.zero_grad()
            # step 2. compute the output
            y_pred = self.model(batch_X)
            # step 3. compute the loss
            loss = self.loss_func(y_pred, batch_y)
            loss_t = loss.item()
            # step 4. use loss to produce gradients
            loss.backward()
            # step 5. use optimizer to take gradient step
            self.optimizer.step()

            train_accuracy = Evaluator.compute_accuracy(y_pred, batch_y)

            training_bar.set_postfix(loss=loss_t, acc=train_accuracy)
            training_bar.update()
        training_bar.close()

    def _val_step(self):
        return Evaluator.compute_accuracy_on_dataset(
            self.model,
            self.dataset,
            "val",
            self.device,
            256 # batch size
        )

    def train(self, args):
        """Do the proper training steps"""
        epoch_bar = tqdm(desc='Epoch',
                          total=self.nr_epochs,
                          position=0,
                          leave=True)

        for epoch_index in range(self.nr_epochs):
            self._train_step()
            val_acc, val_loss = self._val_step()

            strings = list(map(str, [
                self.nr_hidden_neurons, self.nr_filters, self.kernel_length,
                self.learning_rate, self.nr_epochs
            ]))
            filename = "_".join(
                strings + [args.model_state_file]
            )
            filepath = os.path.join(
                args.model_state_dir, filename
            )
            # Save the model
            torch.save(self.model.state_dict(), filepath)
            epoch_bar.set_postfix(val_acc=val_acc, val_loss=val_loss)
            epoch_bar.update()
        epoch_bar.close()



class Evaluator:

    def evaluate_model(self, model, dataset, device):
        print(" Evaluating ".center(80, "="))
        accuracy, loss = Evaluator.compute_accuracy_on_dataset(
            model, dataset, "test", device, 128
        )
        print("Accuracy: {}, Loss: {}".format(accuracy, loss))

    def evaluate_model_from_file(self, filepath, dataset, split, device):
        model, dataset = self._get_model_from_file(filepath)
        self.evaluate_model(
            model, dataset, split, device
        )

    def _get_model_from_file(self, filepath, args):
        """Loads the model from a file. Returns the model and the dataset"""
        filename = os.path.basename(filepath)
        s_hidden, s_filters, s_kernel, *rest = filename.split("_")

        print("Creating the full dataset. This might take a while...")
        dataset = TweetsDataset.load_and_create_dataset(
            args.tweets_fp,
            args.train_dev_fp,
            args.test_fp,
            TrainingRoutine.MAX_TWEET_LENGTH,
        )
        vectorizer = dataset.get_vectorizer()
        vocab_len = len(vectorizer.token_vocab)
        nr_languages = len(vectorizer.label_vocab)
        model = TweetClassifier(
            TrainingRoutine.MAX_TWEET_LENGTH,
            int(s_filters),
            int(s_kernel),
            vocab_len,
            int(s_hidden),
            nr_languages
        )
        # load the weights
        model.load_state_dict(
            torch.load(filepath, map_location=torch.device(args.device))
        )
        model = model.to(args.device)

        return model, dataset

    @staticmethod
    def compute_accuracy(y_pred, y_target):
        _, y_pred_indices = y_pred.max(dim=1)
        n_correct = torch.eq(y_pred_indices, y_target).sum().item()
        return n_correct / len(y_pred_indices) * 100

    @staticmethod
    def compute_accuracy_on_dataset(model, dataset, split, device, batch_size):
        dataset.set_split(split)
        batch_generator = dataset.generate_batches(device=device,
                                                    batch_size=batch_size)
        loss_func = nn.CrossEntropyLoss()
        model.eval()
        accuracy = 0
        total_loss = 0
        for batch_index, (batch_X, batch_y) in enumerate(batch_generator):
            y_pred =  model(x=batch_X)
            loss = loss_func(y_pred, batch_y)
            loss_t = loss.item()
            total_loss += (loss_t - total_loss) / (batch_index + 1)
            acc_t = Evaluator.compute_accuracy(y_pred, batch_y)
            accuracy += (acc_t - accuracy) / (batch_index + 1)

        return accuracy, total_loss


def setup(args):
    """Runtime/Env setup"""
    # create a model directory for saving the model
    if not os.path.exists(args.model_state_dir):
        os.makedirs(args.model_state_dir)

    # Check CUDA and set device
    if not torch.cuda.is_available():
        args.cuda = False

    args.device = torch.device("cuda" if args.cuda else "cpu")
    print("\nUsing CUDA (GPU): {}\n".format(args.cuda))

    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

def get_args():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    to_dir = lambda fp: os.path.join(script_dir, fp)
    return Namespace(
        # Data and Path information
        tweets_fp=to_dir("tweets.json"),
        train_dev_fp=to_dir("labels-train+dev.tsv"),
        test_fp=to_dir("labels-test.tsv"),
        model_state_file="tweet_cnn_model.pth",
        model_state_dir=to_dir("trained_models/"),
        # Runtime Args
        seed=1337,
        cuda=True
    )

def main():
    # Setup
    args = get_args()
    setup(args)

    training = True
    evaluator = Evaluator()
    if training:
        # Train
        training_routine = TrainingRoutine(
            args,
            350,            # nr of filters
            2,              # kernel length
            128,            # nr of hidden neurons
            5,              # nr_epochs
            32,             # batch_size
            0.0001,          # learning_rate
            1,              # data_frac
            0.9,            # train : dev set ratio
        )
        training_routine.start_training_routine(args)
        model = training_routine.model
        dataset = training_routine.dataset
    else:
        # Test
        # Note: only works with models that are trained on the entire dataset
        filename = "128_350_2_0.01_5_tweet_cnn_model.pth"
        model, dataset = evaluator._get_model_from_file(filename, args)

    evaluator.evaluate_model(
        model,
        dataset,
        args.device
    )

# Entrypoint
if __name__ == "__main__":
    main()
