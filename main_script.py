import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import string
import random
import pickle
from transformers import GPT2Tokenizer
from torch.nn.utils.rnn import pad_sequence
from src.model import LyricLSTM
from src.training import train_model
from sklearn.model_selection import train_test_split
import jellyfish


def load_data(load_path):
    with open(load_path, "rb") as file:
        data = pickle.load(file)
    
    return data

def load_model_and_data():
    # Hyperparameter definition
    num_hidden = 512
    num_layers = 4
    embed_size = 200
    drop_prob = 0.3
    lr = 0.0001
    num_epochs = 20
    batch_size = 16

    vocab_size = 50257

    # create the LSTM model
    model = LyricLSTM(num_hidden, num_layers, embed_size, drop_prob, lr,vocab_size)

    # load trained model, make sure to change file path for your machine
    model.load_state_dict(torch.load("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/models/modelpatience7.pt"))
    model.eval()

    # load dictionaries for text generation
    # Again, make sure to change file path for your machine
    word_to_idx = load_data("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/word_to_idx_gpt2.pkl")
    idx_to_word = load_data("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/idx_to_word_gpt2.pkl")

    return model, word_to_idx, idx_to_word

# returns "phonetic disantce" between two words
def nysiis_distance(encoded1,encoded2):
    return jellyfish.hamming_distance(encoded1,encoded2)

# returns NYSIIS phonetic encoding
def nysiis(word):
    return jellyfish.nysiis(word)

def get_soundex_candidate(token, top_tokens, idx_to_word, word_to_idx):
    word = token
    words=[idx_to_word[token] for token in top_tokens]
    # remove gpt2 special character that is not recognised
    if word[0] == "Ġ":
        word = word[1:]
    words = [gpt_word.replace("Ġ","") for gpt_word in words]
    encoded = nysiis(word)
    candidate_nysiis = {candidate: nysiis(candidate) for candidate in words}
    distances = [nysiis_distance(candidate,encoded) for candidate in candidate_nysiis]
    return "Ġ" + words[np.argmax(distances)]


def predict(model, tkn, hidden_layer,idx_to_word,word_to_idx):
         
    # create torch inputs
    x = np.array([[word_to_idx[tkn]]])
    inputs = torch.from_numpy(x).type(torch.LongTensor)

    # detach hidden state from history
    hidden = tuple([layer.data for layer in hidden_layer])

    # get the output of the model
    out, hidden = model(inputs, hidden)

    # get the token probabilities and reshape
    prob = Functional.softmax(out, dim=1).data.numpy()
    prob = prob.reshape(prob.shape[1],)

    # get indices of top 3 values
    top_tokens = prob.argsort()[-5:][::-1]

    # return word and the hidden state
    return top_tokens, hidden

def generate(model, num_words, start_text, max_line_length,idx_to_word,word_to_idx):
    model.eval()
    hidden = model.init_hidden(1)
    tokens = start_text.split()
    generated_lines = []
    i=0
    end_of_line = ""

    for token in start_text.split():
        top_tokens, hidden = predict(model, token, hidden,idx_to_word,word_to_idx)
        tokens.append(idx_to_word[top_tokens[np.random.randint(3)]])


    current_line = tokens.copy()

    for token_num in range(num_words - 1):
        top_tokens, hidden = predict(model, tokens[-1], hidden,idx_to_word,word_to_idx)
        
        token = idx_to_word[top_tokens[np.random.randint(3)]]
        tokens.append(token)

        # Check if adding the token would exceed the maximum line length
        if len(" ".join(current_line)) + len(token) > max_line_length:
            # Start a new line and include the last token
            if i == 0:
                current_line.append(token)
                end_of_line = token
                generated_lines.append(" ".join(current_line))
                current_line = [""]  # Start a new line without including the last token
                i+=1
                
            else:
                # Find the candidate that is the closest phonetically
                current_line.append(get_soundex_candidate(end_of_line,top_tokens,idx_to_word,word_to_idx))
                generated_lines.append(" ".join(current_line))
                current_line = [""]
                i-=1

        else:
            current_line.append(token)  # Add token to the current line

    generated_lines.append(" ".join(current_line))
    return "\n".join(generated_lines)

def edit_lyrics(lyrics):
    lines = lyrics.split('\n')
    edited_lines = []

    for line in lines:
        edited_line=line

        # Remove consecutive repeated words and special characters
        words = re.findall(r'\b\w+\b', edited_line)
        if len(words)==0:
            unique_words=[]
        else:
            unique_words = [words[0]]

        for i in range(1, len(words)):
            if words[i].lower() != words[i - 1].lower():
                unique_words.append(words[i].lower())
        for i in range(len(unique_words)):
            # capitalize the word "i"
            if unique_words[i] == "i":
                unique_words[i] = "I"
        if len(unique_words)!=0:
            # capitalize first word of every line
            unique_words[0]=unique_words[0].capitalize()
        edited_lines.append(unique_words)

    edited_lyrics = '\n'.join([' '.join(line) for line in edited_lines])

    return edited_lyrics



def get_lyric(model, start_text, censor, num_words,idx_to_word,word_to_idx):
    
    # generate the text
    nb_characters_per_line = 40
    generated_text = generate(model, num_words, start_text.lower(),nb_characters_per_line,idx_to_word,word_to_idx)

    # make text look better
    generated_text = generated_text.replace("Ġ","")
    
    return(edit_lyrics(generated_text))