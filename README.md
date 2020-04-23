# Bengali-character-recognition
 How about trying to learn something more complicated than the English alphabet?

Bengali is the official language of the Bangladesh. It has 49 letters in its alphabet with 11 vowels and 38 consonants. In addition, it is also part tonal, with 18 potential diacritics. It has also has an increased grapheme variation, that is the smallest unit in written language. In total, there are around 13K unique character in this language.

The aim of this project is try to classify three constituent of a Bengali handwritten character.

The training data was provided by the Bengali.AI association. It had around 50K handwritten characters and around 6k were unique. All unique character in the dataset were evenly distributed but some component were underrepresented.


# Requirements
 - Keras
 - OpenCV
 - Python Imaging Library

# Usage
Some models weights were split due to a file size limitation. There is a need to use the hsplit executable to join the weights file for the models located in folder "resnet55_multiple" and "resnet30".

- Inference
Put the handwritten images in the examples folder and execute the predict script.
- Training
Only training using data in the parquet format for now. Just put your data in the data folder and launch the training script.
