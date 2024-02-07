import string
import random
from transformers import GPT2Tokenizer
import numpy as np
import re
import pickle
from torch.nn.utils.rnn import pad_sequence
import torch

#Open dataset
file = open("/users/eleves-b/2020/dannel.cassuto/home/Lyrics/all_lyrics.txt", "r", encoding = "utf8")
text = file.read()

#Prepocessing
#Make a list of lyric lines
lyrics = text.lower().split("\n")
lyrics = np.unique(lyrics)[1:].tolist()

# Take out all extra characters
def clean_lyric(lyric):
    return re.sub("[^a-zA-Z' ]", "", lyric).replace("'","")

cleaned_lyrics = [clean_lyric(lyric) for lyric in lyrics]

# Define sequence size
sequence_size = 5

# Create sequences of size sequence_size from a single lyric
def create_sequences(lyric, sequence_len):
    # intialize sequences list
    sequences = []
    
    # returns early if not long enough
    if len(lyric.split()) <= sequence_len:
        return [lyric]
    
    # adds every possible sequence
    for itr in range(sequence_len, len(lyric.split())):
        curr_seq = lyric.split()[itr - sequence_len:itr + 1]
        sequences.append(" ".join(curr_seq))
    
    # returns the sequences
    return sequences


# obtain every sequence
raw_sequences = [create_sequences(lyric, sequence_size) for lyric in cleaned_lyrics]

# filter to get the unique sequences
sequences = np.unique(np.array(sum(raw_sequences, []))).tolist()



''' Bag of words method
uniq_words = np.unique(np.array(" ".join(sequences).split(" ")))
uniq_words_idx = np.arange(uniq_words.size)

word_to_idx = dict(zip(uniq_words.tolist(), uniq_words_idx.tolist()))
idx_to_word = dict(zip(uniq_words_idx.tolist(), uniq_words.tolist()))

vocab_size = len(word_to_idx)

print(vocab_size)

# intialize the empty lists
x_word = []
y_word = []

# iterate through every sequence
for seq in sequences:
    
    # stop if the sequence isn't long enough
    if (len(seq.split()) != sequence_size + 1):
        continue
    
    # add the words to the sequences
    x_word.append(" ".join(seq.split()[:-1]))
    y_word.append(" ".join(seq.split()[1:]))

def get_seq_idx(seq):
    return [word_to_idx[word] for word in seq.split()]

x_idx = np.array([get_seq_idx(word) for word in x_word])
y_idx = np.array([get_seq_idx(word) for word in y_word])
'''

# GPT2 tokenizer method


# Initialize GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Tokenize using GPT-2 tokenizer
tokenized_sequences = [tokenizer.encode(seq, add_special_tokens=True, truncation=True) for seq in sequences]

# Pad or truncate sequences
max_seq_length = 6  
tokenized_sequences = [seq[:max_seq_length] + [0] * (max_seq_length - len(seq)) for seq in tokenized_sequences]

# Convert sequences to numpy arrays
x_idx = np.array([seq[:-1] for seq in tokenized_sequences], dtype=int)
y_idx = np.array([seq[1:] for seq in tokenized_sequences], dtype=int)

# Get gpt2 tokenizer vocabulary
vocab = tokenizer.get_vocab()

# Build idx_to_word and word_to_idx dictionaries
idx_to_word = {idx: token for token, idx in vocab.items()}
word_to_idx = {token: idx for token, idx in vocab.items()}


vocab_size_dataset = len(set(token for seq in tokenized_sequences for token in seq))

print(f"The vocabulary size of your tokenized dataset is: {vocab_size_dataset}")

vocab_size = tokenizer.vocab_size

print(f"The vocabulary size of the GPT-2 tokenizer is: {vocab_size}")

def save_tokenized_data(x_idx,y_idx,word_to_idx,idx_to_word):
    with open("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/x_idx_gpt2.pkl", "wb") as file1:
        pickle.dump(x_idx, file1)
    with open("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/y_idx_gpt2.pkl", "wb") as file2:
        pickle.dump(y_idx, file2)
    with open("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/word_to_idx_gpt2.pkl", "wb") as file3:
        pickle.dump(word_to_idx, file3)
    with open("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/idx_to_word_gpt2.pkl", "wb") as file4:
        pickle.dump(idx_to_word, file4)

save_tokenized_data(x_idx,y_idx,word_to_idx,idx_to_word)