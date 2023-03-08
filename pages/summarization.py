from pathlib import Path
from colorama import Fore
import streamlit as st
import tensorflow as tf
import keras_nlp
from keras import mixed_precision
import keras

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

model_path = Path('summarization\\vocab.txt')


MAX_SEQUENCE_LENGTH = 85
st.header("Summarization Project")
st.subheader("Welcome to my transformer summarization model")

tab, tab1, tab2, tab3 = st.tabs(["Summarization Model", "Model Code and Architecture", "Data cleaning and input", "How to install and run"])

with tab1:
    st.write("This model is a transformer. The arcitecture can be seen here:")
    st.image("https://i.stack.imgur.com/eAKQu.png")
    st.write('This is an extreamly small model. It contains 41,083,221 traininable params. The hyper parameters are as follows:')
    st.code('''BATCH_SIZE = 32
EPOCHS = 10 
MAX_SEQUENCE_LENGTH = 256
VOCAB_SIZE = 50000
EMBED_DIM = 256
INTERMEDIATE_DIM = 512
DOCODER_DIM = 1028
NUM_HEADS = 8

''')
    st.subheader("The model code.")
    st.write("I utilized the keras-nlp libary and tensorflow to create the model arcitecture.")
    st.code('''
encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
)(encoder_inputs)
encoder_outputs = keras_nlp.layers.TransformerEncoder(
    intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS, dropout = 0.2
)(inputs=x)
encoder = keras.Model(encoder_inputs, encoder_outputs)
# Decoder
decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name="decoder_state_inputs")
x = keras_nlp.layers.TokenAndPositionEmbedding(
    vocabulary_size=VOCAB_SIZE,
    sequence_length=MAX_SEQUENCE_LENGTH,
    embedding_dim=EMBED_DIM,
    mask_zero=True,
    
)(decoder_inputs)

x = keras_nlp.layers.TransformerDecoder(
    intermediate_dim=DOCODER_DIM, num_heads=NUM_HEADS, dropout = 0.2
)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(VOCAB_SIZE + 1, activation="softmax")(x)
decoder = keras.Model(
    [
        decoder_inputs,
        encoded_seq_inputs,
    ],
    decoder_outputs,
)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs,
    name="transformer",
)
''')
    st.write("It basically runs like this. Once the input sequence has been cleaned and tokenized (see data cleaning and input), it gets passed as the encoder input. It then gets postionally encoded.")
    st.image('https://machinelearningmastery.com/wp-content/uploads/2022/01/PE3.png')
    st.write("This is a complex process, as seen above with a quick example. If you're interested in how this works this is a pretty good article: https://tinyurl.com/yfeyrre3 \n")
    st.write("After the inputs are positionally embedded, they are passed to the transformer encoder. This is also quite a complicated process, but in the image at the top of this tab you can see the transformer encoder on the left. Four different processes are completed. First, multi-headed attention takes place. What this does is create a matrix where the entire input can get passed into a dense network at one time. After normalizing and adding in the orginal input tokens, we push the inputs through a dense network. Once the matrix has been outputted by the dense network, we add and normalize again and the encoder has finished its job.")
    st.write("Now things get a little more complicated. One step forward the expected outputs get passed into a multi-headed attention. After adding and normalization another multiheaded attention takes place, this time utilizing the encoder outputs. This then gets passed through another dense network before two activation functions are applied to the output and finally the output matrix is returned.")
with tab2:
    st.subheader("Before data can be inputed into the model it must be cleaned and tokenized")
    st.markdown("Data is tokenized by splitting up each word in the given input. If we take, for example, the phrase: `('I am a dog')`. It would change to `['I'], ['am'], ['a'], ['dog']` and be returned by the tokenizer.")
    st.markdown("The tokenizer takes a certian vocabulary, in this case of size 50,000, and does a few things to the input. First if the word isn't in the vocabulary file it will replace it with an `['UNK']` token. It will also lowcase every word and strip all of the accent marks as well. We define the tokenizer using the keras-nlp:")
    st.code('''tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary='vocab.txt file',
    lowercase=True, 
    strip_accents=True)
''')
    st.markdown("After the input has been tokenized it is necessary to do a start end pack. What this means is: at the start and end of every token sequence place a start and end token. This looks like: `['START'], ['I'], ['am'], ['a'], ['dog'], ['END'].` In some cases the input sequences might be shorter than the input sequence length. If this is the case we do a process called \"padding\". When a token under goes padding it gets extended. If we use the \"I am a dog\" example, but the largetest input was \"I really love it when I pet dogs\" we would extend the length of the \"I am a dog\" example by four `['PAD']` tokens. This would change the final output to be `['START'], ['I'], ['am'], ['a'], ['dog'], ['PAD'], ['PAD'], ['PAD'], ['PAD'], ['END']`. This is initialized via:")
    st.code('''token_ids_packer = keras_nlp.layers.StartEndPacker(
    start_value=tokenizer.token_to_id("[START]"),
    end_value=tokenizer.token_to_id("[END]"),
    pad_value=tokenizer.token_to_id("[PAD]"),
    sequence_length=MAX_SEQUENCE_LENGTH,
)
    ''')
with tab3:
    st.subheader("How to install and run!")
    st.write("So things are going to get a bit weird, but first things first clone this github repository: https://github.com/dashblack576/seq2seqV1.git. This is the model file. The model weights and biases though, will have to be downloaded and put into the seq2seqV1 file.")
    st.write("Once this is cloned, and the model file has been dragged inside, place the project folder into the summarization file. Once this is done, click the rerun button in the upper right hand corner.")

with tab:
    if(model_path.is_file()):

        text = st.text_input("Input the text you want summarized here:", max_chars= 10000)
        max_return_length = st.slider("How long do you want the summarization to be?", min_value=20, max_value=MAX_SEQUENCE_LENGTH)


        transformer = keras.models.load_model('summarization\\Model')

        tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
            vocabulary='summarization\\vocab.txt',
            lowercase=True,
            strip_accents=True,
        )


        def decode_sequences(input_sentences):
            batch_size = tf.shape(input_sentences)[0]

            # Tokenize the encoder input.
            encoder_input_tokens = tokenizer(input_sentences).to_tensor(
                shape=(None, MAX_SEQUENCE_LENGTH)
            )

            # Define a function that outputs the next token's probability given the
            # input sequence.
            def token_probability_fn(decoder_input_tokens):
                return transformer([encoder_input_tokens, decoder_input_tokens])[:, -1, :]

            # Set the prompt to the "[START]" token.
            prompt = tf.fill((batch_size, 1), tokenizer.token_to_id("[START]"))

            generated_tokens = keras_nlp.utils.greedy_search(
                token_probability_fn,
                prompt,
                max_length=max_return_length,
                end_token_id=tokenizer.token_to_id("[END]"),
            )
            generated_sentences = tokenizer.detokenize(generated_tokens)
            return generated_sentences


        if("a" or "e" or "i" or "o" or "u" in text):
            summerized = decode_sequences(tf.constant([text]))
            summerized = summerized.numpy()[0].decode("utf-8")
            summerized = (
                summerized.replace("[PAD]", "")
                    .replace("[START]", "")
                    .replace("[END]", "")
                    .replace("[UNK]", "")
                    .strip()
                )
            st.write("\n \n \n" + summerized)

    else:
        st.write(':RED[It does not appear as though you have the model downloaded. Please download it and place it into the summarization folder. It can be found here: https://github.com/dashblack576/seq2seqV1]')