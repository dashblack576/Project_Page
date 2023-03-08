import tensorflow as tf
from datasets import load_dataset
import random
import keras
import keras_nlp
from keras import mixed_precision




policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

MAX_SEQUENCE_LENGTH = 256

#test_ds = load_dataset('xsum', split='test')


test_ds = load_dataset('cnn_dailymail', '3.0.0', split='test')

test_ds = test_ds.rename_column("article", "document")
test_ds = test_ds.rename_column("highlights", "summary")

transformer = keras.models.load_model('C:/Users/dashb/Documents/capstoneProject/seq2seqV1/tmp/model_checkpoint')

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary='C:/Users/dashb/Documents/capstoneProject/seq2seqV1/vocab.txt',
    lowercase=True,
    strip_accents=True
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
        max_length=40,
        end_token_id=tokenizer.token_to_id("[END]"),
    )
    generated_sentences = tokenizer.detokenize(generated_tokens)
    return generated_sentences

test_pair = random.choice(test_ds)
test = test_pair['document']
translated = decode_sequences(tf.constant([test]))
translated = translated.numpy()[0].decode("utf-8")
translated = (
    translated.replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .replace("[UNK]", "")
        .replace("#", "")
        .strip()
    )
print("\n" + test)
print("\n" + test_pair['summary'])
print("\n \n \n" + translated)