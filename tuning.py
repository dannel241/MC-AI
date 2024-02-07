from itertools import product
from src.training import train_model_early_stopping
from src.model import LyricLSTM
from joblib import Parallel, delayed
import pickle
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch



def load_data(load_path):
    with open(load_path, "rb") as file:
        data = pickle.load(file)
    
    return data



def evaluate_model(parameters):

    x_idx = load_data("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/x_idx_gpt2.pkl")
    y_idx = load_data("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/y_idx_gpt2.pkl")
    word_to_idx = load_data("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/word_to_idx_gpt2.pkl")
    idx_to_word = load_data("/users/eleves-b/2020/dannel.cassuto/MC_AI/data/idx_to_word_gpt2.pkl")

    x_train, x_val, y_train, y_val = train_test_split(x_idx, y_idx, test_size=0.2, random_state=42)

    # Define hyperparameter values that are not tested by grid search
    num_layers = 4
    embed_size = 200
    num_epochs = 2
    vocab_size = 50257
    # Instantiate model with the given hyperparameters
    model = LyricLSTM(parameters['num_hidden'], num_layers, embed_size, parameters['drop_prob'], parameters['lr'], vocab_size)
    
    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['lr'])

    # Train the model on training data and validate
    accuracy = train_model_early_stopping(model, optimizer, loss_func, x_train, y_train,x_val,y_val, num_epochs=num_epochs, batch_size=parameters['batch_size'], patience =1,num_hidden = parameters['num_hidden'])
    print('Model with parameters ', parameters['num_hidden'], parameters['drop_prob'], parameters['lr'], parameters['batch_size'], " has an accuracy of ", accuracy)
    return accuracy




# Perform grid search
best_accuracy = 0.0
best_hyperparameters = {}

# define different hyperparameters values to test
param_grid = {
    'num_hidden': [128, 256, 512],
    'drop_prob' : [0.2, 0.3, 0.4],
    'lr' : [0.001, 0.01, 0.1],
    'batch_size' : [16, 32, 64],
}

# Generate all possible combinations of hyperparameters
parameter_combinations = [{'num_hidden': p1, 'drop_prob': p2, 'lr' : p3, 'batch_size' : p4} for p1 in param_grid['num_hidden'] for p2 in param_grid['drop_prob'] for p3 in param_grid['lr'] for p4 in param_grid['batch_size']]

# Set the number of parallel jobs
num_jobs = -1  # Use all available cores

# Perform parallel grid search
results = Parallel(n_jobs=num_jobs)(delayed(evaluate_model)(params) for params in parameter_combinations)

# Find the best hyperparameters based on the results
best_params = parameter_combinations[results.index(max(results))]
best_accuracy = max(results)

# display best parameters and best accuracy
print("Best Hyperparameters:", best_params)
print("Best Accuracy:", best_accuracy)