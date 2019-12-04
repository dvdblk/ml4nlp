import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import datetime
from torch.utils.data import Dataset, DataLoader
from argparse import Namespace
from tqdm import tqdm


class Vocabulary:
    UNK_TOKEN="UNKNOWN"

    def __init__(self, add_unk=True):
        self._token_to_ids = {}
        self._ids_to_token = {}
        # Add the unknown token and index
        if add_unk:
            self.unk_index = self.add_token(Vocabulary.UNK_TOKEN)
        else:
            self.unk_index = -1

    def add_token(self, token):
        if token in self._token_to_ids:
            index = self._token_to_ids[token]
        else:
            index = len(self._token_to_ids)
            self._token_to_ids[token] = index
            self._ids_to_token[index] = token
        return index

    def lookup_token(self, token):
        return self._token_to_ids.get(token, self.unk_index)

    def lookup_index(self, index):
        if index not in self._ids_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._ids_to_token[index]

    def __len__(self):
        return len(self._token_to_ids)

class Vectorizer:

    def __init__(self, token_vocab, label_vocab):
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self._nr_of_tokens = len(self.token_vocab)

    def vectorize(self, sentence):
        sentences_ids = [self.token_vocab.lookup_token(t) for t in sentence]
        return torch.tensor(sentences_ids, dtype=torch.long)

    def vectorize_labels(self, tags):
        tags_ids = [self.label_vocab.lookup_token(t) for t in tags]
        return torch.tensor(tags_ids, dtype=torch.long)

class NERDataset(Dataset):
    
    INPUT_X="x_data"
    TARGET_Y="y_target"

    def __init__(self, train_df, dev_df, test_df, vectorizer):
        super(NERDataset, self).__init__()
        self._lookup_dict = {'train' : train_df,
                             'dev': dev_df,
                             'test': test_df}
        self._vectorizer = vectorizer

        self.set_split()

    @classmethod
    def load_and_create_dataset(cls, train_fp, dev_fp, test_fp, vocab,
        data_frac=1):
        # Data sets, no preprocessing needed
        train, tag_vocab = NERDataset._read_csv(train_fp)
        dev, _ = NERDataset._read_csv(dev_fp)
        test, _ = NERDataset._read_csv(test_fp)
        vectorizer = Vectorizer(vocab, tag_vocab)
        return cls(train, dev, test, vectorizer)

    @staticmethod
    def _read_csv(filepath):
        tokens = []
        bio_tags = []
        tag_set = set()
        with open(filepath, mode="r") as f:
            # Accumulators for the data
            sentence_tokens = [] # words in a sentence
            sentence_bio_tags = [] # IOB/BIO tags of words in the sentence
            # Create a generator for the lines of the file without empty lines (\n)
            line_gen = (line.rstrip('\n') for line in f if line != "\n")

            for line in line_gen:
                if line[0] != "#":
                    # Accumulation case, i.e. not a '#' line
                    # *_ is used to catch an edge case where the line has 
                    # more than 3 '\t' characters...
                    _, token, bio_outer_tag, *_ = tuple(line.split('\t'))
                    sentence_tokens.append(token)
                    sentence_bio_tags.append(bio_outer_tag)
                elif sentence_tokens and sentence_bio_tags:
                    # '#' line case, except for the very first one
                    # add the new data instance to the full lists, and reset 
                    # the accumulators
                    tokens.append(sentence_tokens)
                    bio_tags.append(sentence_bio_tags)
                    tag_set.update(sentence_bio_tags)
                    sentence_tokens, sentence_bio_tags = [], []
        # Create the dataframe
        df = pd.concat([
            pd.Series(tokens, name="Tokens"),
            pd.Series(bio_tags, name="Tags")
        ], axis=1)
        # Create the tag vocabulary
        tag_vocab = Vocabulary(add_unk=False)
        for tag in tag_set:
            tag_vocab.add_token(tag)
        return df, tag_vocab

    def get_vectorizer(self):
        """returns the vectorizer"""
        return self._vectorizer

    def set_split(self, split = "train"):
        """selects the splits in the dataset"""
        self._target_df = self._lookup_dict[split]

    def __len__(self):
        return len(self._target_df)

    def __getitem__(self, index):
        vectorizer = self.get_vectorizer()
        row = self._target_df.iloc[index]

        sentence_vector = vectorizer.vectorize(row.Tokens)
        tag_vector = vectorizer.vectorize_labels(row.Tags)
        return { NERDataset.INPUT_X: sentence_vector, 
                 NERDataset.TARGET_Y: tag_vector }

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
            # yield the batch_X and batch_target_y
            yield (
                out_data_dict[NERDataset.INPUT_X],
                out_data_dict[NERDataset.TARGET_Y]
            )

class RNN(nn.Module):
    def __init__(self, embeddings, n_tags):
        super(RNN, self).__init__()
        # maps each token to an embedding_dim vector
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        lstm_hidden_dim = 50
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, n_tags)


    def forward(self, inputs):
        #apply the embedding layer that maps each token to its embedding
        out = self.embedding(inputs)   # dim: batch_size x batch_max_len x embedding_dim

        #run the LSTM along the sentences of length batch_max_len
        out, _ = self.lstm(out)     # dim: batch_size x batch_max_len x lstm_hidden_dim                

        #reshape the Variable so that each row contains one token
        out = out.view(-1, out.shape[2])  # dim: batch_size*batch_max_len x lstm_hidden_dim

        #apply the fully connected layer and obtain the output for each token
        out = self.fc(out)          # dim: batch_size*batch_max_len x num_tags

        return nn.functional.log_softmax(out, dim=1)   # dim: batch_size*batch_max_len x num_tags


# Create token vocab / embeddings
def load_embedding_and_create_vocab(filepath):
    embeddings = np.empty((0, 100), dtype=np.float64)
    vocab = Vocabulary()
    with open(filepath, mode="r") as f:
        for line in f:
            token, *embedding = line.rstrip().split(" ")
            embedding = [np.float64(x) for x in embedding]
            vocab.add_token(token)
            embeddings = np.append(embeddings, [embedding], axis=0)

    return embeddings, vocab


class TrainingRoutine:
    """Encapsulates the training"""

    def __init__(self, args, dataset, embeddings, lstm_hidden_dim,
                nr_epochs, batch_size, learning_rate):
        # Params
        self.dataset = dataset
        self.lstm_hidden_dim = lstm_hidden_dim
        self.nr_epochs = nr_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = args.device

        # Model
        vectorizer = self.dataset.get_vectorizer()
        n_tags = len(vectorizer.label_vocab)
        model = RNN(embeddings, n_tags)
        self.model = model.to(args.device)
        # Loss with class weights
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

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
            "Hidden Dim {}, LR {}".format(
                self.lstm_hidden_dim, self.learning_rate
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

            # Progress bar updates
            bar.set_postfix_str("Loss={1:.3f}".format(loss_t))
            bar.update()

    def _val_step(self):
        self.dataset.set_split('dev')
        # Create the batch generator
        batch_generator = dataset.generate_batches(device=self.device,
                                                    batch_size=batch_size)
        loss_func = nn.CrossEntropyLoss()
        model.eval()
        total_loss = 0
        # Compute loss / accuracy over the entire set
        for batch_index, (batch_X, batch_y) in enumerate(batch_generator):
            y_pred =  model(x=batch_X)
            loss = loss_func(y_pred, batch_y)
            loss_t = loss.item()
            total_loss += (loss_t - total_loss) / (batch_index + 1)

        return total_loss

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
            train_fp=to_dir("tweets.json"),
            dev_fp=to_dir("labels-train+dev.tsv"),
            test_fp=to_dir("labels-test.tsv"),
            model_state_file="lstm_model.pth",
            model_state_dir=to_dir("trained_models/"),
            # Runtime Args
            seed=1337,
            cuda=True,
            training=True
        )

if __name__ == "__main__":
    # Setup
    args = Util.get_args()
    Util.setup(args)

    emb, vocab = load_embedding_and_create_vocab("data/small.vocab")

    embeddings_tensor = torch.tensor(emb, dtype=torch.float64).to(args.device)

    dataset = NERDataset.load_and_create_dataset(
        "data/NER-de-train.tsv",
        "data/NER-de-dev.tsv",
        "data/NER-de-test.tsv",
        vocab
    )

    training_routine = TrainingRoutine(args, dataset, embeddings_tensor,
        lstm_hidden_dim=50,
        nr_epochs=5, 
        batch_size=64, 
        learning_rate=0.01
    )
    training_routine.start_training_routine(args)

