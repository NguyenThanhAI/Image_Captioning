import os
import argparse

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

    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=299)
    parser.add_argument("--sequence_length", type=int, default=20)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    parser.add_argument("--recurrent_type", type=str, default="lstm")

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

    #print("PREDICTED CAPTION:", end=" ")
    #print(decoded_caption.replace("<start> ", "").replace(" <end>", "").strip())
    return decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()


if __name__ == '__main__':
    args = get_args()

    image_path = args.image_path
    model_dir = args.model_dir
    config_file = args.config_file
    image_size = args.image_size
    sequence_length = args.sequence_length
    ff_dim = args.ff_dim
    embed_dim = args.embed_dim
    dropout_rate = args.dropout_rate
    recurrent_type = args.recurrent_type

    vectorization, num_vocabs = get_text_vectorizer(config_file=config_file, sequence_length=20, text_data=None)

    print("Num vocabularies: {}".format(num_vocabs))

    cnn_model = get_cnn_model(image_size=image_size)

    encoder = RecurrentEncoder(hidden_dim=embed_dim, dropout_rate=dropout_rate, type=recurrent_type)

    decoder = RecurrentDecoder(embed_dim=embed_dim, dropout_rate=dropout_rate, ff_dim=ff_dim, type=recurrent_type,
                               hidden_dim=embed_dim, vocab_size=num_vocabs)

    caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
    caption_model.built = True
    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = sequence_length - 1

    generate_caption(sample_img=image_path, image_size=image_size)

    caption_model.load_weights(os.path.join(model_dir, "caption_model_best.h5"))

    caption = generate_caption(sample_img=image_path, image_size=image_size)
    print("PREDICTED CAPTION:", end=" ")
    print(caption)
