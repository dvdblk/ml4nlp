import torch
import pandas as pd

def _read_csv(filepath):
    tokens = []
    bio_tags = []
    with open(filepath, mode="r") as f:
        # Accumulators for the data
        sentence_tokens = [] # words in a sentence
        sentence_bio_tags = [] # IOB/BIO tags of words in the sentence
        # Create a generator for the lines of the file without empty lines (\n)
        line_gen = (line.rstrip('\n') for line in f if line != "\n")

        for line in line_gen:
            # Split the line into the values
            if line[0] != "#":
                # Accumulation case, i.e. not a '#' line
                # *_ is used to catch an edge case where the line has 
                # more than 3 '\t' characters...
                _, token, bio_outer, bio_embed, *_ = tuple(line.split('\t'))
                sentence_tokens.append(token)
                sentence_bio_tags.append((bio_outer, bio_embed))
            elif    sentence_tokens and sentence_bio_tags:
                # '#' line case, except for the very first one
                # add the new data instance to the full lists, and reset 
                # the accumulators
                tokens.append(sentence_tokens)
                bio_tags.append(sentence_bio_tags)
                sentence_tokens, sentence_bio_tags = [], []
    # Create the dataframe
    return pd.concat([
        pd.Series(tokens, name="Tokens"),
        pd.Series(bio_tags, name="BIO tags")
    ], axis=1)



train = _read_csv("data/NER-de-train.tsv")
dev = _read_csv("data/NER-de-dev.tsv")
test = _read_csv("data/NER-de-test.tsv")


