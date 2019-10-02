# -*- coding: utf-8 -*-
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
import string
lines= pd.read_csv('./fra-eng/fra.txt', names=['src', 'tar'], sep='\t')
# print('line length', len(lines))
# print(lines.src.tolist()[0:100])
src_input_sentences = lines.src.tolist()
tar_output_sentences = lines.tar.apply(lambda x: x + ' <eos>').tolist()
tar_input_sentencse = lines.tar.apply(lambda x : '<sos> ' + x + ' <eos>').tolist()

print('src_input_sentences', src_input_sentences[1000:1005])
print('tar_output_sentences', tar_output_sentences[1000:1005])
print('tar_input_sentencse', tar_input_sentencse[1000:1005])

src_tokenizer = Tokenizer()
tar_tokenizer = Tokenizer()
src_tokenizer.fit_on_texts(src_input_sentences)
tar_tokenizer.fit_on_texts(tar_input_sentencse)
src_vocab_size = len(src_tokenizer.word_index) + 1
tar_vocab_size = len(tar_tokenizer.word_index) + 1

# print(src_vocab_size)

src_encoded = src_tokenizer.texts_to_sequences(src_input_sentences)
tar_encoded = tar_tokenizer.texts_to_sequences(tar_input_sentencse)
tar_decoded = tar_tokenizer.texts_to_sequences(tar_output_sentences)

max_src_len=max(len(l) for l in src_encoded)
max_tar_len=max(len(l) for l in tar_encoded)

encoder_input = pad_sequences(src_encoded, max_src_len, padding='post')
decoder_input = pad_sequences(tar_encoded, max_tar_len, padding='post')
decoder_target = pad_sequences(tar_decoded, max_tar_len, padding='post')

print('encoder_input', encoder_input[1000:1005])
print('decoder_input', decoder_input[1000:1005])
print('decoder_target', decoder_target[1000:1005])
