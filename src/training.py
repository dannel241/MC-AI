import torch
import torch.nn as nn
import torch.nn.functional as Functional
import numpy as np
from tqdm import tqdm

#model_path = "/users/eleves-b/2020/dannel.cassuto/MC_AI/data/models/"


'''Initial training function with no validation'''
def train_model(model, optimizer, loss_func, x_idx, y_idx, num_epochs, batch_size, model_path):
    model.train()
    for epoch in range(num_epochs):
        print("epoch number ", epoch)
        # initialize hidden state
        hidden_layer = model.init_hidden(batch_size)
            
        for x, y in get_next_batch(x_idx, y_idx, batch_size):
                
            # convert numpy arrays to PyTorch arrays
            inputs = torch.from_numpy(x).type(torch.LongTensor)
            act = torch.from_numpy(y).type(torch.LongTensor)

            # reformat the hidden layer
            hidden_layer = tuple([layer.data for layer in hidden_layer])

            # obtain the zero-accumulated gradients from the model
            model.zero_grad()
                
            # get the output from the model
            output, hidden = model(inputs, hidden_layer)
                
            # calculate the loss from this prediction
            loss = loss_func(output, act.view(-1))

            # back-propagate to update the model
            loss.backward()

            # prevent exploding gradient problem
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            # update weigths using the optimizer
            optimizer.step()  
    
    # Save the model
    torch.save(model.state_dict(), model_path)

''' Actual training function that implements validation and early stopping '''
def train_model_early_stopping(model, optimizer, loss_func, x_train, y_train, x_val, y_val, num_epochs, batch_size, patience,num_hidden, model_path):
    model.train()
    
    best_val_loss = np.inf
    current_patience = 0

    # use tdqm to get estimates of run time
    for epoch in tqdm(range(num_epochs), "training epoch "):
        
        # initialize hidden state
        hidden_layer = model.init_hidden(batch_size)
        correct_predictions = 0
        total_samples = 0

        for x, y in get_next_batch(x_train, y_train, batch_size):
                
            # convert numpy arrays to PyTorch arrays
            inputs = torch.from_numpy(x).type(torch.LongTensor)
            act = torch.from_numpy(y).type(torch.LongTensor)

            # reformat the hidden layer
            hidden_layer = tuple([layer.data for layer in hidden_layer])

            # obtain the zero-accumulated gradients from the model
            model.zero_grad()
                
            # get the output from the model
            output, hidden = model(inputs, hidden_layer)
                
            # calculate the loss from this prediction
            loss = loss_func(output, act.view(-1))

            # back-propagate to update the model
            loss.backward()

            # prevent exploding gradient problem
            nn.utils.clip_grad_norm_(model.parameters(), 1)

            # update weights using the optimizer
            optimizer.step() 

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in get_next_batch(x_val, y_val, batch_size):
                val_inputs = torch.from_numpy(x_batch).type(torch.LongTensor)
                val_act = torch.from_numpy(y_batch).type(torch.LongTensor)

                val_hidden_layer = model.init_hidden(x_batch.shape[0])
                val_output, _ = model(val_inputs, val_hidden_layer)

                val_loss += loss_func(val_output, val_act.view(-1)).item()
                # Calculate accuracy
                predictions = torch.argmax(val_output, dim=-1)
                correct_predictions += torch.sum(predictions == val_act.view(-1))
                total_samples += len(val_act.view(-1))

        # Average validation loss
        val_loss /= len(x_val) // batch_size

        # Calculate accuracy
        accuracy = correct_predictions.item() / total_samples
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Accuracy: {accuracy}')

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            current_patience = 0
        else:
            current_patience += 1

        # Early stopping
        if current_patience >= patience:
            print(f'Early stopping after {epoch + 1} epochs.')
            break

    # Save the model

    torch.save(model.state_dict(), model_path)

    #print(f'Model with parameters {model.parameters} has an accuracy of {accuracy}')
    return accuracy

def get_next_batch(x, y, batch_size):
    
    # iterate until the end of x
    for itr in range(batch_size, x.shape[0], batch_size):
        
        # obtain the indexed x and y values
        batch_x = x[itr - batch_size:itr, :]
        batch_y = y[itr - batch_size:itr, :]
        
        # yield these values
        yield batch_x, batch_y
