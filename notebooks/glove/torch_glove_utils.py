from __future__ import division

import os
import logging

from collections import Counter, defaultdict
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(format='%(asctime)s - %(levelname)s- %(message)s', level=logging.INFO)
logger  = logging.getLogger(__name__)

class NotFitToCorpusError(Exception):
    pass

class Vocab(object):
    def __init__(self, freqs, max_vocab_size, min_occurrences):
        """Vocabulary object

        Args:
            freqs (dict): count dictionary with frequency information
            max_vocab_size (int): max vocabulary size
            min_occurrences (int): min vocabulary count
        """
        # freqs
        self.freqs = freqs
        self.max_vocab_size = max_vocab_size
        self.min_occurrences = min_occurrences
        # itos index to word list
        self.itos = [token for token, count in freqs.most_common(self.max_vocab_size)
                        if count >= self.min_occurrences]
        # words to index dict
        self.stoi = {token: i for i, token in enumerate(self.itos)}

    def __len__(self):
        """Return len of vocabulary

        Returns:
            len of vocabulary (int)
        """
        return len(self.itos)

    def __getitem__(self, word_str_or_id):
        if isinstance(word_str_or_id, str):
            return self.stoi[word_str_or_id]
        elif isinstance(word_str_or_id, int):
            return self.itos[word_str_or_id]

class GloveDataset(Dataset):
    def __init__(self, context_size):
        """Glove dataset, this module is in charge of fitting the dataset and create 
        a coocurrence matrix.

        Args:
            docs (list): list of list of tokens previously prepared.
            context_size (int): windows size of context.
        """

        if isinstance(context_size, tuple):
            self.left_context, self.right_context = context_size
        elif isinstance(context_size, int):
            self.left_context = self.right_context = context_size
        else:
            raise ValueError("`context_size` should be an int or a tuple of two ints")

    def fit_to_corpus(self, docs, max_vocab_size=1000000, min_occurrences=1):
        """fit and generate coocurrence matrix for glove embedding.

        Args:
            vocab_size (int): size of vocabulary.
            min_occurrences (int): minimun number of occurrences.
            concurrency_cap (int): max concurrency
            docs (list): list of list of tokens
        """
        # frequency dictionary for tokens
        freqs = Counter()
        # Coocurrence counts
        coocurrence_counts = defaultdict(float)
        logger.info("Building coocurrence frequency dictionary distance weights")

        it = 0
        for tokens in docs:
            # Update frequency dictionary
            freqs.update(tokens)
            for left_context, token, right_context in self.__context_windows__(tokens, self.left_context, self.right_context):
                for i, context_token in enumerate(left_context[::-1]):
                    coocurrence_counts[(token, context_token)] += 1 / (i + 1)
                for i, context_token in enumerate(right_context):
                    coocurrence_counts[(token, context_token)] += 1 / (i + 1)
            if it%100 == 0:
                logger.info("Progress: %i" %(it))
            it += 1
            
        if len(coocurrence_counts) == 0:
            raise ValueError("No  coccurrences in corpus")
        
        self.vocab = Vocab(freqs, max_vocab_size, min_occurrences)
        # coocurrence matrix
        self.coocurrence_matrix = [
            (self.vocab.stoi[_tokens[0]], self.vocab.stoi[_tokens[1]], count)
            for _tokens, count in coocurrence_counts.items()
            if _tokens[0] in self.vocab.stoi and _tokens[1] in self.vocab.stoi
            ]

    def __context_windows__(self, tokens, left_size, right_size):
        """Yield generator for context windows.

        Args:
            tokens (list): list of tokens
            left_size (int): left_size windows
            right_size (int): right size windows

        Yields:
            (left_context, token, right_context) (tuple): windows tokens
        """
        for i, token in enumerate(tokens):
            start_index = i - left_size
            end_index = i + right_size
            left_context = self.__window__(tokens, start_index, i - 1)
            right_context = self.__window__(tokens, i + 1, end_index)
            yield (left_context, token, right_context)

    def __window__(self, tokens, start_index, end_index):
        """Returns the list of words starting from `start_index`, going to `end_index`
        taken from region. If `start_index` is a negative number, or if `end_index`
        is greater than the index of the last word in region, this function will pad
        its return value with `NULL_WORD`.

        Args:
            tokens (list): A list of tokens
            start_index (int): The start index of the window
            end_index (int): The end index of the window
        """
        last_index = len(tokens) + 1
        selected_tokens = tokens[max(start_index, 0):min(end_index, last_index) + 1]
        return selected_tokens


    def __len__(self):
        if not hasattr(self, 'coocurrence_matrix'):
            raise NotFitToCorpusError("Not building the corpus")
        return len(self.coocurrence_matrix)

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets
        
        Args:
            index (int): the index to the data point 
        Returns:
            a dictionary holding the data point's:
                features (token, context_token)
                label (count)
        """
        if not hasattr(self, 'coocurrence_matrix'):
            raise NotFitToCorpusError("Need to fit model to corpus before")

        token, context_token, count = self.coocurrence_matrix[index]

        return {
            "token": token,
            "context_token": context_token,
            "count": count
        }

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset
        
        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size

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

