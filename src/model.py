import torch
import torch.nn as nn
import torch.nn.functional as Functional

class LyricLSTM(nn.Module):
    
    ''' Initialize the network variables '''
    def __init__(self, num_hidden, num_layers, embed_size, drop_prob, lr,vocab_size):
        # call super() on the class
        super().__init__()
        
        # store the constructor variables
        self.drop_prob = drop_prob
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.lr = lr
        
        # define the embedded layer
        self.embedded = nn.Embedding(vocab_size, embed_size, padding_idx = 0)

        # define the LSTM
        self.lstm = nn.LSTM(embed_size, num_hidden, num_layers, dropout = drop_prob, batch_first = True)
        
        # define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # define the fully-connected layer
        self.fc = nn.Linear(num_hidden, vocab_size)      
    
    ''' Forward propogate through the network '''
    def forward(self, x, hidden):
        
        ## pass input through embedding layer
        embedded = self.embedded(x)     
        
        # Obtain the outputs and hidden layer from the LSTM layer
        lstm_output, hidden = self.lstm(embedded, hidden)
        
        # pass through a dropout layer and reshape
        dropout_out = self.dropout(lstm_output).reshape(-1, self.num_hidden) 

        ## put "out" through the fully-connected layer
        out = self.fc(dropout_out)

        # return the final output and the hidden state
        return out, hidden
    
    ''' Initialize the hidden state of the network '''
    def init_hidden(self, batch_size = 32):
        
        # Create a weight torch using the parameters of the model
        weight = next(self.parameters()).data

        # initialize the hidden layer using the weight torch
        hidden = (weight.new(self.num_layers, batch_size, self.num_hidden).zero_(),
                  weight.new(self.num_layers, batch_size, self.num_hidden).zero_())
        
        # return the hidden layer
        return hidden