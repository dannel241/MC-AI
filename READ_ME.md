
- **src:** Contains the code for model, dataset tokenization, lyrics scrapping and training the model
- **data:** Holds the different trained models and lyric databases.

Feel free to adjust the structure based on your needs.

## Getting Started

### Prerequisites

Usual Python libraries: numpy, torch, pickle, sklearn.

Need transformers library, gpt2 tokenizer used.

User interface built with tkinter library.

### Installation

Download repository. Pretrained models have an acceptable performance but more training is required.

Make sure to change file paths to be able to run on your machine. You only need to modify them in main_script.py, where the model is loaded and the data as well. They are saved in the project, all that is needed is to provide save path to the project.

### Parameters

Change number of lines in app.py: total_number_of_words in function on_generate_button_click

Change length of lines : nb_characters_per_line in main_script.py, get_lyric fuction.


