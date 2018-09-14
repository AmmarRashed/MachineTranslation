# MachineTranslation
Seq2Seq model for machine translation using TensorFlow
- Special thanks to <a href="https://towardsdatascience.com/@parkchansung"> Park Chansung</a> for his awesome <a href="https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f">tutorial</a>
<img src="https://github.com/AmmarRashed/MachineTranslation/blob/master/seq2seq.png?raw=true" alt="Image from Aurélien Géron's Hands-on Machine Learning with Scikit-learn and Tensorflow">

## Data
- Retrieved from <a href="https://github.com/deep-diver/EN-FR-MLT-tensorflow/tree/master/data"> deep-diver</a>
- 137,861 pair of sentences
- English vocab size: 199
- French vocab size:  339

## Trained model
<img src="https://github.com/AmmarRashed/MachineTranslation/blob/master/stats.png?raw=true">

## Usage
- Training:
  - `train("Data/en", "Data/fr", is_tar=False)`  
  or if you wish to extract from within the script:
  - `tar = tarfile.open("Data/en-fr_small.tar.gz" , "r:gz")`
  - `train(source_file_name, target_file_name, tar="Data/en-fr_small.tar.gz", is_tar=True)`
- `train` Function parameters:
  - `source_file`:Source filename
  - `target_file`:Target filename
  - `max_num_sentences=None`:Maximum number of sentences to take from dataset
  - `max_vocab_size_source=None`:Maximum vocab size of the source language
  - `max_vocab_size_target=None`:Maximum vocab size of the target language
  - `preprocess_pathos.path.join("checkpoints","preprocess.p")`:Path to save the preprocessed data in as a checkpoint
  - `validation_set_size=None`:Validation set size (Default is min of 5% of the sentences or 1000)
  - `epochs=10`: Number of epochs
  - `batch_size=128`: Batch size
  - `rnn_size=128`: The number of units in the GRU cell
  - `num_layers=3`: The number of layers of GRU cells
  - `encoding_embedding_size=200`: Embedding size used in the encoder
  - `decoding_embedding_size=200`: Embedding size used in the decoder
  - `learning_rate=0.001`: Learning rate (alpha)
  - `keep_probability=0.5`: Keep probability used in dropout regularization
  - `reverse_source=True`: If `True`, the order of words in the source sentence is reversed but not the target sentences
  - `max_display_step=300`: Printing training stats either 3 times per epoch or every `max_display_step` batchs in each epoch
  - `save_path = os.path.join("checkpoints","dev")`: Path of saving the model
  - `is_tar=False`: Is the data file provided a tar file?
    - Requires `tar` parameter
  - `tar=None`: The tar file path
  
- Translating
  - `input_sentence = "I love french. it is a nice language."`
  - `translated_sentence = translate(input_sentence)`
