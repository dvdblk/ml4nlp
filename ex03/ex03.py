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

class Vocabulary:
    UND_TOKEN="und"

    def __init__(self):
        self._token_to_ids = {}
        self._ids_to_token = {}

        # Add the unknown token and index
        self.und_index = self.add_token(Vocabulary.UND_TOKEN)

    def vocabulary_set(self):
        """this function returns a list of unique tokens"""
        return list(self._ids_to_token.values())

    def add_token(self, token):
        """Update mapping dicts based on the token."""
        if token in self._token_to_ids:
            index = self._token_to_ids[token]
        else:
            index = len(self._token_to_ids)
            self._token_to_ids[token] = index
            self._ids_to_token[index] = token
        return index

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the 'und' index if token isn't present.
        """
        return self._token_to_ids.get(token, self.und_index)

    def lookup_index(self, index):
        """Return the token associated with the index"""
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

        return cls(tokens_vocab, labels_vocab, max_twt_length)


class TweetsDataset(Dataset):

    INPUT_X="x_data"
    TARGET_Y="y_target"
    MAX_TWEET_LENGTH=256

    def __init__(self, dataframes):
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
            self.train_df,
            TweetsDataset.MAX_TWEET_LENGTH
        )

        # Add labels in alphabetical order
        labels_groupped = self.train_df.groupby([COL_LABEL]).agg(["count"])
        sorted_labels = labels_groupped.index.tolist()
        for label in sorted_labels:
            self._vectorizer.label_vocab.add_token(label)
        # Label frequencies
        freq = labels_groupped.values
        und_index = self._vectorizer.label_vocab.lookup_token(
            Vocabulary.UND_TOKEN
        )
        freq = np.insert(
            np.ndarray.flatten(freq).astype(float), und_index, 1e-7
        )
        freq = torch.tensor(freq, dtype=torch.float32)
        # Most frequent label
        max_label_freq = float(freq.max().item())
        # Set the class weights
        self.class_weights = torch.mul(torch.reciprocal(freq), max_label_freq)
        # Prepare for training
        self.set_split()

    @classmethod
    def load_and_create_dataset(cls, tweets_fp, train_fp, test_fp,
        data_frac=1, train_dev_frac=0.9):
        """
        Args: filepath
        """
        tweets = TweetsDataset._get_tweets(tweets_fp)
        train_labels = TweetsDataset._get_train_labels(train_fp)
        test_labels = TweetsDataset._get_test_labels(test_fp)
        data = TweetsDataset._create_sets(
            tweets, train_labels, test_labels, data_frac, train_dev_frac,
        )
        return cls(data)

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

        only_keep_balanced_languages = True
        if only_keep_balanced_languages:
            # Option 1 - keep the rows that are labelled with a balanced language
            train_dev_labels = train_dev_labels[balanced_labels]
        else:
            # Option 2 - replace rows that are labelled with an imbalanced language
            # ~ is element-wise logical not
            train_dev_labels.loc[~balanced_labels, COL_LABEL] = 'und'
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
                nr_hidden, dropout_p, nr_languages):
        super(TweetClassifier, self).__init__()
        maxpool_kernel_size = min(kernel_length*10, input_length - 1)
        # Convolution layer
        self.conv1 = torch.nn.Conv1d(one_hot_length, nr_filters, kernel_length)
        # Pooling
        self.pool = torch.nn.MaxPool1d(maxpool_kernel_size)
        # Dropout
        self.dropout = torch.nn.Dropout(p=dropout_p)
        # Fully Connected layers
        fc1_in_len = int(input_length/maxpool_kernel_size)*nr_filters
        self.fc1 = torch.nn.Linear(fc1_in_len, nr_hidden)
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
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        #print("fc1 (shape: {}): {}".format(out.shape, out))
        out = F.relu(self.fc2(out))
        #print("fc2/output (shape: {}): {}".format(out.shape, out))
        return out

class TrainingRoutine:
    """Encapsulates the training of a classifier"""

    def __init__(self, args, dataset, nr_filters, kernel_length, nr_hidden,
                nr_epochs, batch_size, learning_rate, dropout_p):
        self.dataset = dataset
        self.nr_hidden_neurons = nr_hidden
        self.nr_filters = nr_filters
        self.kernel_length = kernel_length
        self.nr_epochs = nr_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_p = dropout_p
        self.device = args.device

        # Model
        vectorizer = self.dataset.get_vectorizer()
        vocab_len = len(vectorizer.token_vocab)
        nr_languages = len(vectorizer.label_vocab)

        model = TweetClassifier(
            input_length=dataset.MAX_TWEET_LENGTH,
            nr_filters=nr_filters,
            kernel_length=kernel_length,
            one_hot_length=vocab_len,
            nr_hidden=nr_hidden,
            dropout_p=dropout_p,
            nr_languages=nr_languages
        )
        self.model = model.to(args.device)
        self.loss_func = nn.CrossEntropyLoss(
            weight=self.dataset.class_weights.to(self.device)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[5, 10], gamma=0.1
        )

    def start_training_routine(self, args):
        """Create and train a training routine

        Returns:
            training_routine (TrainingRoutine): the trained routine
        """
        start = time.time()
        self.dataset.set_split('train')
        nr_batches = len(self.dataset) // self.batch_size
        print(" Training ".center(Util.NCOLS, "="))
        print(Util.print_time_fmt(
            "Started training {} epochs with {} batches of size {}.".format(
                self.nr_epochs, nr_batches, self.batch_size
            )
        ))
        print(Util.print_time_fmt(
            "Filters {}, Neurons {}, Kernel {}, LR {}, Dropout {}".format(
                self.nr_filters, self.nr_hidden_neurons, self.kernel_length,
                self.learning_rate, self.dropout_p
            )
        ))
        # Create the progress bar
        progress_bar = tqdm(desc='', total=self.nr_epochs * nr_batches,
            leave=False, ncols=Util.NCOLS
        )
        # Start the training
        self.train(args, nr_batches, progress_bar)
        end = time.time()
        progress_bar.close()
        print(Util.print_time_fmt(
            "Finished training in {0:.{1}f} seconds.".format(end - start, 2)
        ))

    def _train_step(self, bar, epoch, batches):
        """Do a training iteration over the batch"""
        # setup: batch generator, set train mode on
        self.dataset.set_split('train')
        batch_generator = self.dataset.generate_batches(device=self.device,
                                                    batch_size=self.batch_size)

        # make sure our weights / embeddings get updated
        self.model.train()

        for batch_index, (batch_X, batch_y) in enumerate(batch_generator):
            bar.set_description_str("E {} | B {}".format(
                epoch, batch_index
            ))
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

            bar.set_postfix_str("Acc={0:.2f}, Loss={1:.3f}".format(
                train_accuracy, loss_t
            ))
            bar.update()

    def _val_step(self):
        return Evaluator.compute_accuracy_on_dataset(
            self.model,
            self.dataset,
            "val",
            self.device,
            256 # batch size
        )

    def train(self, args, nr_batches, progress_bar):
        """Do the proper training steps and update the progress bar"""
        for epoch_index in range(self.nr_epochs):
            # Train step
            self._train_step(progress_bar, epoch_index+1, nr_batches)
            # Validation step
            progress_bar.display(Util.print_time_fmt(
                "Epoch {}: Performing validation step...".format(
                    epoch_index+1
                ))
            )
            val_acc, val_loss = self._val_step()

            # Scheduler updates the learning rate if needed
            self.scheduler.step()

            # Model filename
            strings = list(map(str, [
                self.nr_hidden_neurons, self.nr_filters, self.kernel_length,
                self.learning_rate, self.nr_epochs
            ]))
            filename = "_".join(strings + [args.model_state_file])
            filepath = os.path.join(args.model_state_dir, filename)
            # Save the model
            torch.save(self.model.state_dict(), filepath)

            # Update the progress bar
            progress_bar.write(Util.print_time_fmt(
                "Epoch {}: {}".format(
                    epoch_index+1, Util.accuracy_loss_fmt(val_acc, val_loss)
                )
            ))
            progress_bar.refresh()


class Evaluator:

    def evaluate_model(self, model, dataset, device):
        print(" Evaluating ".center(Util.NCOLS, "="))
        accuracy, loss = Evaluator.compute_accuracy_on_dataset(
            model, dataset, "test", device, 128
        )
        print(Util.print_time_fmt(
            "Test set: {}".format(Util.accuracy_loss_fmt(accuracy, loss))
        ))

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
            args.test_fp
        )
        vectorizer = dataset.get_vectorizer()
        vocab_len = len(vectorizer.token_vocab)
        nr_languages = len(vectorizer.label_vocab)
        model = TweetClassifier(
            TweetsDataset.MAX_TWEET_LENGTH,
            int(s_filters),
            int(s_kernel),
            vocab_len,
            int(s_hidden),
            1, # we don't need dropout while testing
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

class Util:
    """Utility functions"""

    SEPARATOR = "_"
    NCOLS = 80

    @staticmethod
    def print_time_fmt(str):
        return "[{0:%Y-%m-%d %H:%M:%S}] {1}".format(datetime.datetime.now(), str)

    @staticmethod
    def accuracy_loss_fmt(acc, loss):
        return "Accuracy {0:.2f}%, Loss {1:.3f}".format(acc, loss)

    @staticmethod
    def get_model_filename(nr_hidden,
        learning_rate, nr_epochs, generic_filename="model.pth"):
        strings = list(map(str, [
            nr_hidden, learning_rate, nr_epochs
        ]))
        return FSUtil.SEPARATOR.join(
            strings + [generic_filename]
        )

    @staticmethod
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

    @staticmethod
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
            cuda=True,
            training=True
        )

def main():
    # Setup
    args = Util.get_args()
    Util.setup(args)

    # Create the dataset
    dataset = TweetsDataset.load_and_create_dataset(
        args.tweets_fp,
        args.train_dev_fp,
        args.test_fp,
        1,              # fraction of data to use, used for debugging
        0.9                 # train to dev set ratio
    )
    # Train || Test
    if args.training:
        # Train
        routines = []
        routines.append(TrainingRoutine(args, dataset,
            350,            # nr of filters
            2,              # kernel length
            128,            # nr of hidden neurons
            3,              # nr_epochs
            32,             # batch_size
            0.001,          # learning_rate
            0.5             # dropout
        ))
        routines_params = [
            [300, 2, 128, 3, 32, 0.001, 0.5],
            [300, 3, 128, 3, 32, 0.001, 0.5],
            [300, 3, 128, 3, 64, 0.001, 0.5],
            [250, 2, 128, 3, 32, 0.001, 0.5],
            [250, 2, 128, 3, 64, 0.001, 0.5],
            [350, 4, 128, 3, 32, 0.001, 0.5],
            [350, 4, 128, 3, 64, 0.001, 0.5],
            [350, 2, 128, 3, 64, 0.001, 0.6],
            [350, 2, 128, 3, 64, 0.001, 0.7],
        ]
        for routine_params in routines_params:
            routines.append(TrainingRoutine(args, dataset, *routine_params))

        # Evaluate
        evaluator = Evaluator()
        for routine in routines:
            routine.start_training_routine(args)
            evaluator.evaluate_model(
                routine.model, routine.dataset, args.device
            )
    else:
        # Test
        # Note: only works with models that are trained on the entire dataset
        filename = "128_350_2_0.01_5_tweet_cnn_model.pth"
        model, dataset = evaluator._get_model_from_file(filename, args)
        Evaluator().evaluate_model(
            model,
            dataset,
            args.device
        )

# Entrypoint
if __name__ == "__main__":
    main()
