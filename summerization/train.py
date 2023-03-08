import tensorflow as tf
import keras_nlp
import keras
from datasets import load_dataset
from keras import mixed_precision



policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)




gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    #Restrict Tensorflow to only allocate 7gb of memory on the first GPU
   try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0],
       [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=7168)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
       #virtual devices must be set before GPUs have been initialized
        print(e)

BATCH_SIZE = 32
MAX_SEQUENCE_LENGTH = 256
VOCAB_SIZE = 50000


train_ds, val_ds = load_dataset('cnn_dailymail', '3.0.0', split=['train','validation'])

train_ds, val_ds = train_ds.rename_column("article", "document"), val_ds.rename_column("article", "document")
train_ds, val_ds = train_ds.rename_column("highlights", "summary"), val_ds.rename_column("highlights", "summary")


def encode(example):

    example['document'] = example['document'].encode('ascii','ignore')
    example['summary'] = example['summary'].encode('ascii','ignore')

    return example

train_ds = train_ds.map(encode)
val_ds = val_ds.map(encode)

tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary='C:/Users/dashb/Documents/capstoneProject/seq2seqV1/vocab.txt',
    lowercase=True,
    strip_accents=True,
)

token_ids_packer = keras_nlp.layers.StartEndPacker(
    start_value=tokenizer.token_to_id("[START]"),
    end_value=tokenizer.token_to_id("[END]"),
    pad_value=tokenizer.token_to_id("[PAD]"),
    sequence_length=MAX_SEQUENCE_LENGTH,
)

target_ids_packer = keras_nlp.layers.StartEndPacker(
    pad_value=tokenizer.token_to_id("[PAD]"),
    sequence_length=MAX_SEQUENCE_LENGTH + 1,
)

def preprocess_batch(document, summary):
    batch_size = tf.shape(summary)[0]

    document = tokenizer(document)
    summary = tokenizer(summary)

    document = token_ids_packer(document)
    summary = target_ids_packer(summary)


    return (
        {
            "encoder_inputs": document,
            "decoder_inputs": summary[:, :-1],
        },
        summary[:, 1:],
    )


def make_dataset(data):
    doc = data['document']
    summ = data['summary']


    dataset = tf.data.Dataset.from_tensor_slices((doc, summ))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.shuffle(512).prefetch(16)


train_ds = make_dataset(train_ds)
val_ds = make_dataset(val_ds)





transformer = keras.models.load_model("C:/Users/dashb/Documents/capstoneProject/seq2seqV1/Model")

opt = tf.keras.optimizers.RMSprop(
    learning_rate=0.0001,
    decay=0.000001,
    name="RMSprop",
    epsilon=1e-07,
    clipvalue=1
)

transformer.summary()

transformer.compile(
    optimizer = opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

checkpoint = ('C:\\Users\\dashb\\Documents\\capstoneProject\\seq2seqV1\\tmp\\model_checkpoint')

callback = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=6, verbose=1),
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1, verbose=1, min_lr = 0.00001, patience=4),
            tf.keras.callbacks.ModelCheckpoint(checkpoint)
]

for x in range(4): 
    transformer.fit(train_ds, epochs = 8, validation_data=val_ds, callbacks=[callback])
    transformer.save("C:/Users/dashb/Documents/capstoneProject/seq2seqV1/Model", save_traces = True)
    print(x + 1)
