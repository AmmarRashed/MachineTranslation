# MachineTranslation
Seq2Seq model for machine translation using TensorFlow
- Special thanks to <a href="https://towardsdatascience.com/@parkchansung"> Park Chansung</a> for his awesome <a href="https://towardsdatascience.com/seq2seq-model-in-tensorflow-ec0c557e560f">tutorial</a>
<img src="https://camo.githubusercontent.com/d7b33a62f27a21ebd8b0e13e64b4051fdd89735d/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f3830302f312a5f7253484c6a4653686b6e4175336a74337262634e512e706e67">

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
  - `tar = tarfile.open("Data/en-fr_small.tar.gz", "r:gz")`
  - `train(source_file_name, target_file_name, is_tar=True, tar=tar)`
- Translating
  - `input_sentence = "I love french. it is a nice language."`
  - `translated_sentence = translate(input_sentence)`
