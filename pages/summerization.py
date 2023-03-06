import streamlit as st
import tensorflow as tf
import keras_nlp
from keras import mixed_precision
import keras
from pathlib import Path


policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

MAX_SEQUENCE_LENGTH = 85

st.header("Summerization Project")
st.subheader("Welcome to my transformer summerization model")

st.write('This is an extreamly small model. It contains about 25,000,000 traininable params. The hyper parameters are as follows:')
st.code('''BATCH_SIZE = 176
EPOCHS = 10
MAX_SEQUENCE_LENGTH = 85
VOCAB_SIZE = 30000
EMBED_DIM = 256
INTERMEDIATE_DIM = 512
DOCODER_DIM = 1024
NUM_HEADS = 8''')

tab1, tab2, tab3 = st.tabs(["Model Code and Architecture", "Data cleaning and input", "How to install and run"])

with tab1:
    st.write("This model is a transformer. The arcitecture can be seen here:")
    st.image("https://i.stack.imgur.com/eAKQu.png")
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

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
    st.subheader("How to install and run!")
    st.write("So things are going to get a bit weird, but first things first clone this github repository: https://github.com/dashblack576/seq2seqV1.git. This is the model file. The model weights and biases though, will have to be downloaded and put into the seq2seqV1 file.")
    st.write("Once this is cloned, and the model file has been dragged inside, place the project folder into the summerization file. Once this is done, click the rerun button in the upper right hand corner.")
   




model_path = Path('summerization\\seq2seqV1\\vocab.txt')

if(model_path.is_file()):
    text = st.text_input("Input the text you want summerized here:", max_chars= 10000)




    transformer = keras.models.load_model('summerization\\seq2seqV1\\Model')

    tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
        vocabulary='summerization\\seq2seqV1\\vocab.txt',
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

        generated_tokens = keras_nlp.utils.top_p_search(
            token_probability_fn,
            prompt,
            p=0.1,
            max_length=40,
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
        st.text("\n \n \n" + summerized)


