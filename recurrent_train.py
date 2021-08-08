import os
import argparse

import pickle

from tqdm import tqdm

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from recurrent_models import get_cnn_model, RecurrentEncoder, RecurrentDecoder, ImageCaptioningModel
from dataset_utils import load_captions_data, train_val_split, get_text_vectorizer, save_text_vectorizer, make_dataset, \
    read_image


seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str, default=None)
    parser.add_argument("--val_images_dir", type=str, default=None)
    parser.add_argument("--caption_path", type=str, default=None)
    parser.add_argument("--val_caption_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--sequence_length", type=int, default=30)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=2048)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    parser.add_argument("--recurrent_type", type=str, default="lstm")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--data_file", type=str, default=None)

    args = parser.parse_args()

    return args


def generate_caption(sample_img, image_size):
    # Select a random image from the validation dataset
    #sample_img = np.random.choice(valid_images)

    # Read the image from the disk
    sample_img = read_image(sample_img, size=(image_size, image_size))
    img = sample_img.numpy().astype(np.uint8)
    plt.imshow(img)
    plt.show()

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        t_decoded_caption = tf.convert_to_tensor(decoded_caption)[tf.newaxis]
        tokenized_caption = vectorization(t_decoded_caption)[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token

    print("PREDICTED CAPTION:", end=" ")
    print(decoded_caption.replace("<start> ", "").replace(" <end>", "").strip())


if __name__ == '__main__':
    args = get_args()

    images_dir = args.images_dir
    val_images_dir = args.val_images_dir
    caption_path = args.caption_path
    val_caption_path = args.val_caption_path
    save_dir = args.save_dir
    image_size = args.image_size
    sequence_length = args.sequence_length
    ff_dim = args.ff_dim
    embed_dim = args.embed_dim
    batch_size = args.batch_size
    buffer_size = args.buffer_size
    num_epochs = args.num_epochs
    dropout_rate = args.dropout_rate
    recurrent_type = args.recurrent_type
    config_file = args.config_file
    data_file = args.data_file

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Load the dataset

    # Split the dataset into training and validation sets
    if os.path.basename(caption_path).startswith("captions") and os.path.basename(caption_path).endswith(".json"):
        if not os.path.exists(data_file):
            train_data, train_text_data = load_captions_data(filename=caption_path,
                                                             images_dir=images_dir)
            valid_data, val_text_data = load_captions_data(filename=val_caption_path,
                                                           images_dir=val_images_dir)
            text_data = train_text_data + val_text_data

            with open(data_file, "wb") as f:
                pickle.dump({"train": train_data, "val": valid_data, "text_data": text_data}, f)
                f.close()
            print("Save data to file")
        else:
            with open(data_file, "rb") as f:
                data = pickle.load(f)
                train_data = data["train"]
                valid_data = data["val"]
                text_data = data["text_data"]
                f.close()
            print("Load data from file")
            keys = list(train_data.keys())

            for key in tqdm(keys):
                #train_data[os.path.join(images_dir, os.path.basename(key))] = train_data.pop(key)
                train_data[os.path.join(images_dir, os.path.split(key)[1])] = train_data.pop(key)

            keys = list(valid_data.keys())

            for key in tqdm(keys):
                #valid_data[os.path.join(val_images_dir, os.path.basename(key))] = valid_data.pop(key)
                valid_data[os.path.join(val_images_dir, os.path.split(key)[1])] = valid_data.pop(key)
    else:
        captions_mapping, text_data = load_captions_data(filename=caption_path,
                                                         images_dir=images_dir)
        train_data, valid_data = train_val_split(captions_mapping)
    print("Number of training samples: ", len(train_data))
    print("Number of validation samples: ", len(valid_data))

    if config_file is None:
        print("Build and save tokenizer")
        vectorization, num_vocabs = get_text_vectorizer(config_file=None, sequence_length=sequence_length,
                                                        text_data=text_data)

        save_text_vectorizer(vectorization=vectorization, config_file=os.path.join(save_dir, "tokenizer.pkl"))
    else:
        print("Load tokenizer")
        vectorization, num_vocabs = get_text_vectorizer(config_file=config_file, sequence_length=sequence_length,
                                                        text_data=None)

    print("Num vocabularies: {}".format(num_vocabs))

    train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()), vectorization=vectorization,
                                 image_size=image_size, batch_size=batch_size, buffer_size=buffer_size)
    valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()), vectorization=vectorization,
                                 image_size=image_size, batch_size=batch_size, buffer_size=buffer_size)

    cnn_model = get_cnn_model(image_size=image_size)

    encoder = RecurrentEncoder(hidden_dim=embed_dim, dropout_rate=dropout_rate, type=recurrent_type)

    decoder = RecurrentDecoder(embed_dim=embed_dim, dropout_rate=dropout_rate, ff_dim=ff_dim, type=recurrent_type,
                               hidden_dim=embed_dim, vocab_size=num_vocabs)

    caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)

    cross_entropy = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_acc")

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.8,
                                                    patience=2, min_lr=1e-6, verbose=True)

    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(save_dir, "caption_model_best.h5"),
                                                    monitor="val_acc", verbose=1, save_best_only=True,
                                                    save_weights_only=True, mode="max")

    caption_model.compile(optimizer=keras.optimizers.Adam(), loss=cross_entropy)

    caption_model.fit(train_dataset,
                      epochs=num_epochs,
                      validation_data=valid_dataset,
                      callbacks=[early_stopping, reduce_lr, checkpoint])

    print("Finished training")

    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = sequence_length - 1
    valid_images = list(valid_data.keys())

    for _ in range(10):
        sample_img = np.random.choice(valid_images)
        generate_caption(sample_img=sample_img, image_size=image_size)
