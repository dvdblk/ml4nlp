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
    PAD_TOKEN="PADDING"

    def __init__(self, add_unk=True):
        self._token_to_ids = {}
        self._ids_to_token = {}
        # Add the unknown token and index
        if add_unk:
            self.unk_index = self.add_token(Vocabulary.UNK_TOKEN)
        else:
            self.unk_index = -1

    @classmethod
    def init_from_dict(cls, idx_to_token):
        vocab = Vocabulary(add_unk=False)
        vocab._ids_to_token = idx_to_token
        vocab._token_to_ids = { v:k for k,v in idx_to_token.items() }
        vocab.unk_index = vocab.add_token(Vocabulary.UNK_TOKEN)
        return vocab

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
        return self._ids_to_token.get(index, Vocabulary.UNK_TOKEN)

    def __len__(self):
        return len(self._token_to_ids)

class Vectorizer:

    def __init__(self, token_vocab, label_vocab):
        self.token_vocab = token_vocab
        self.label_vocab = label_vocab
        self._nr_of_tokens = len(self.token_vocab)

    def vectorize(self, sentence):
        sentences_ids = [self.token_vocab.lookup_token(t) for t in sentence]
        return np.array(sentences_ids, dtype=np.int64)

    def vectorize_labels(self, tags):
        tags_ids = [self.label_vocab.lookup_token(t) for t in tags]
        return np.array(tags_ids, dtype=np.int64)

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
                                shuffle=shuffle, drop_last=drop_last,
                                collate_fn=self._collate_fn_padd)

        for data_dict in dataloader:
            out_data_dict = {}
            for name, tensor in data_dict.items():
                out_data_dict[name] = tensor.to(device)
            # yield the batch_X and batch_target_y
            yield (
                out_data_dict[NERDataset.INPUT_X],
                out_data_dict[NERDataset.TARGET_Y]
            )

    def _collate_fn_padd(self, batch):
        # get a np array from a list of dicts in the format
        # { 'x': [...], 'y': [...] }, np_batch is of size `batch_size`
        np_batch = pd.DataFrame(batch).values
        X = np_batch[:, 0] # sentences
        Y = np_batch[:, 1] # labels
        batch_size = len(X)
        # get the lengths of sentences in this batch
        lengths = [len(sentence) for sentence in X]
        vectorizer = self.get_vectorizer()
        pad_token = Vocabulary.PAD_TOKEN
        pad_sentence_idx = vectorizer.token_vocab.lookup_token(pad_token) # 1
        pad_labels_idx = vectorizer.label_vocab.lookup_token(pad_token) # -1
        longest_sentence = max(lengths)
        padded_X = np.ones((batch_size, longest_sentence)) * pad_sentence_idx
        padded_Y = np.ones((batch_size, longest_sentence)) * pad_labels_idx

        for i, sent_len in enumerate(lengths):
            sentence = X[i]
            tags = Y[i]
            padded_X[i, 0:sent_len] = sentence[:sent_len]
            padded_Y[i, 0:sent_len] = tags[:sent_len]
        return { NERDataset.INPUT_X: torch.tensor(padded_X, dtype=torch.long), 
                 NERDataset.TARGET_Y: torch.tensor(padded_Y, dtype=torch.long) }


class RNN(nn.Module):
    def __init__(self, embeddings, n_tags, sentence_padding_idx):
        super(RNN, self).__init__()
        self.n_tags = n_tags
        # maps each token to an embedding_dim vector
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding.from_pretrained(
            embeddings.float(), 
            padding_idx=sentence_padding_idx
        )
        lstm_hidden_dim = 50
        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_dim, n_tags)


    def forward(self, inputs):
        batch_size, longest_sentence_len = inputs.shape

        #apply the embedding layer that maps each token to its embedding
        # dim: batch_size x batch_max_len x embedding_dim
        out = self.embedding(inputs)   
    
        #run the LSTM along the sentences of length batch_max_len
        # dim: batch_size x batch_max_len x lstm_hidden_dim
        out, _ = self.lstm(out)                     
        out = out.contiguous()

        #reshape the Variable so that each row contains one token
        # dim: batch_size*batch_max_len x lstm_hidden_dim
        out = out.view(-1, out.shape[2])  

        #apply the fully connected layer and obtain the output for each token
        # dim: batch_size*batch_max_len x num_tags
        out = self.fc(out)          

        # dim: batch_size*batch_max_len x num_tags
        out = nn.functional.log_softmax(out, dim=1)   

        return out

    def loss(self, y_pred, y_true):
        #y_pred = torch.argmax(y_pred, axis=2).float()

        y_true = y_true.view(-1)

        mask = (y_true >= 0).float()

        y_true = y_true % y_pred.shape[1]

        n_tokens = int(torch.sum(mask))

        return -torch.sum(y_pred[range(y_pred.shape[0]), y_true]*mask)/n_tokens


# Create token vocab / embeddings
def load_embedding_and_create_vocab(filepath):
    embeddings = np.empty((0, 100), dtype=np.float64)
    vocab = Vocabulary()
    print(Util.print_time_fmt("Loading the embeddings..."))
    with open(filepath, mode="r") as f:
        line_count = len(f.readlines())
        progress_bar = tqdm(
            desc='', total=line_count, leave=False, ncols=Util.NCOLS
        )
        f.seek(0, 0)
        for line in f:
            token, *embedding = line.rstrip().split(" ")
            embedding = [np.float64(x) for x in embedding]
            vocab.add_token(token)
            embeddings = np.append(embeddings, [embedding], axis=0)
            progress_bar.update()
    progress_bar.close()
    print(Util.print_time_fmt("Loading finished."))
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
        padding_idx = vectorizer.token_vocab.lookup_token(Vocabulary.PAD_TOKEN)
        tag_padding_idx = vectorizer.label_vocab.lookup_token(Vocabulary.PAD_TOKEN)
        model = RNN(embeddings, n_tags, padding_idx)
        self.model = model.to(args.device)
        self.loss_func = model.loss
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
            bar.set_postfix_str("Loss={0:.3f}".format(loss_t))
            bar.update()

    def _val_step(self):
        self.dataset.set_split('dev')
        # Create the batch generator
        batch_generator = dataset.generate_batches(device=self.device,
                                                    batch_size=self.batch_size)
        self.model.eval()
        accuracy = 0
        total_loss = 0
        # Compute loss / accuracy over the entire set
        for batch_index, (batch_X, batch_y) in enumerate(batch_generator):
            y_pred =  self.model(batch_X)
            loss = self.loss_func(y_pred, batch_y)
            loss_t = loss.item()
            total_loss += (loss_t - total_loss) / (batch_index + 1)
            acc_t = self.accuracy(y_pred, batch_y)
            accuracy += (acc_t - accuracy) / (batch_index + 1)

        return accuracy, total_loss

    def accuracy(self, outputs, labels):
        # reshape labels to give a flat vector of length batch_size*seq_len
        labels = labels.view(-1)

        # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
        mask = (labels >= 0)

        # np.argmax gives us the class predicted for each token by the model
        outputs = torch.argmax(outputs, axis=1)

        # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
        result = torch.sum(outputs == labels)/torch.sum(mask).float()
        return result

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

            # Model filename
            strings = list(map(str, [
                self.lstm_hidden_dim, self.learning_rate, self.nr_epochs
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
        return "Accuracy {0:.2f}%, Loss {1:.3f}".format(acc*100, loss)

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

    emb, vocab = load_embedding_and_create_vocab("data/medium.vocab")

    embeddings_tensor = torch.tensor(emb, dtype=torch.float64).to(args.device)

    dataset = NERDataset.load_and_create_dataset(
        "data/NER-de-train.tsv",
        "data/NER-de-dev.tsv",
        "data/NER-de-test.tsv",
        vocab
    )

    # Train
    training_routine = TrainingRoutine(args, dataset, embeddings_tensor,
        lstm_hidden_dim=100,
        nr_epochs=15, 
        batch_size=64, 
        learning_rate=0.01
    )
    training_routine.start_training_routine(args)

    # Eval
    training_routine.dataset.set_split('test')
    # Create the batch generator
    batch_generator = dataset.generate_batches(device=args.device,
                                                batch_size=training_routine.batch_size,
                                                shuffle=False)
    training_routine.model.eval()
    results_file = open("results.txt", mode="w")
    tag_vocab = training_routine.dataset.get_vectorizer().label_vocab
    for batch_index, (batch_X, batch_y) in enumerate(batch_generator):
        y_pred =  training_routine.model(batch_X)
        loss = training_routine.loss_func(y_pred, batch_y)
        y_pred = torch.argmax(y_pred, axis=1)
        y_pred = y_pred.reshape(batch_y.shape)
        for i, (y, y_true, token) in enumerate(zip(y_pred, batch_y, batch_X)):
            n_row = batch_index * training_routine.batch_size + i
            row = training_routine.dataset._target_df.iloc[n_row]
            for n_word, (pred_tag, true_tag, word_idx) in enumerate(zip(y, y_true, token)):
                if true_tag == -1:
                    # ignore padding
                    break
                results_file.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(
                    n_word+1, 
                    row.Tokens[n_word],
                    tag_vocab.lookup_index(true_tag.item()),
                    "O",
                    tag_vocab.lookup_index(pred_tag.item()),
                    "O"
                ))
    results_file.close()

