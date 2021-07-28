import os

import re

import pickle

from typing import Tuple, List
from itertools import groupby

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def load_captions_data(filename, images_dir: str):
    """Loads captions (text) data and maps them to corresponding images.

    Args:
        filename: Path to the text file containing caption data.

    Returns:
        caption_mapping: Dictionary mapping image names and the corresponding captions
        text_data: List containing all the available captions
    """
    if os.path.basename(filename) == "Flickr8k.token.txt":
        with open(filename) as caption_file:
            caption_data = caption_file.readlines()
            caption_mapping = {}
            text_data = []

            for line in caption_data:
                line = line.rstrip("\n")
                # Image name and captions are separated using a tab
                img_name, caption = line.split("\t")
                # Each image is repeated five times for the five different captions. Each
                # image name has a prefix `#(caption_number)`
                img_name = img_name.split("#")[0]
                img_name = os.path.join(images_dir, img_name.strip())

                if img_name.endswith("jpg"):
                    # We will add a start and an end token to each caption
                    caption = "<start> " + caption.strip() + " <end>"
                    text_data.append(caption)

                    if img_name in caption_mapping:
                        caption_mapping[img_name].append(caption)
                    else:
                        caption_mapping[img_name] = [caption]
    elif os.path.basename(filename) == "result.csv":
        df = pd.read_csv(filename, sep='|', header=None)
        records = df.to_records(index=False)
        records = list(records)[1:]
        captions_mapping = {}
        text_data = []
        # "<start> " + caption.strip() + " <end>"
        for image, captions in groupby(records, key=lambda x: x[0]):
            caption_list = list(captions)
            # print(type(caption_list[0][2]), caption_list[0][2])
            caption_list = list(map(lambda x: "<start> " + str(x[2]).strip() + " <end>", caption_list))
            captions_mapping[os.path.join(images_dir, image)] = caption_list
            text_data.extend(caption_list)
    else:
        raise ValueError("Unknown dataset")

    return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.8, shuffle=True):
    """Split the captioning dataset into train and validation sets.

    Args:
        caption_data (dict): Dictionary containing the mapped caption data
        train_size (float): Fraction of all the full dataset to use as training data
        shuffle (bool): Whether to shuffle the dataset before splitting

    Returns:
        Traning and validation datasets as two separated dicts
    """

    # 1. Get the list of all image names
    all_images = list(caption_data.keys())

    # 2. Shuffle if necessary
    if shuffle:
        np.random.shuffle(all_images)

    # 3. Split into training and validation sets
    train_size = int(len(caption_data) * train_size)

    training_data = {
        img_name: caption_data[img_name] for img_name in all_images[:train_size]
    }
    validation_data = {
        img_name: caption_data[img_name] for img_name in all_images[train_size:]
    }

    # 4. Return the splits
    return training_data, validation_data


def custom_standardization(input_string):
    strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
    strip_chars = strip_chars.replace("<", "")
    strip_chars = strip_chars.replace(">", "")
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


#vectorization = TextVectorization(
#    max_tokens=None,
#    output_mode="int",
#    output_sequence_length=20,
#    standardize=custom_standardization,
#)
#vectorization.adapt(text_data)
#
#num_vocabs = len(vectorization.get_vocabulary())
#
#print("num_vocabs: {}".format(num_vocabs))
#print(vectorization.get_vocabulary())


def get_text_vectorizer(config_file: str, sequence_length: int, text_data: List[str]) -> Tuple[TextVectorization, int]:
    if config_file is not None:
        with open(config_file, "rb") as f:
            config = pickle.load(f)
        vectorization = TextVectorization.from_config(config["config"])
        vectorization.set_weights(config["weights"])
    else:
        vectorization = TextVectorization(
            max_tokens=None,
            output_mode="int",
            output_sequence_length=sequence_length,
            standardize=custom_standardization,
        )
        vectorization.adapt(tf.data.Dataset.from_tensor_slices(text_data)) # Must convert list of string to tf dataset

    num_vocabs = len(vectorization.get_vocabulary())

    return vectorization, num_vocabs


def save_text_vectorizer(vectorization: TextVectorization, config_file: str) -> None:
    with open(config_file, "wb") as f:
        pickle.dump({"config": vectorization.get_config(),
                     "weights": vectorization.get_weights()}, f)


def read_image(img_path, size: Tuple[int, int]):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size=size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def make_dataset(images, captions, vectorization: TextVectorization,
                 image_size: int, batch_size: int, buffer_size=2056) -> tf.data.Dataset:
    img_dataset = tf.data.Dataset.from_tensor_slices(images).map(
        lambda x: read_image(x, size=(image_size, image_size)), num_parallel_calls=tf.data.AUTOTUNE
    )
    cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(
        vectorization, num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


## Load the dataset
#captions_mapping, text_data = load_captions_data(filename=r"F:\Flickr8k_text\Flickr8k.token.txt",
#                                                 images_dir=r"F:\Flickr8k_Dataset\Flicker8k_Dataset")
#
## Split the dataset into training and validation sets
#train_data, valid_data = train_val_split(captions_mapping)
#print("Number of training samples: ", len(train_data))
#print("Number of validation samples: ", len(valid_data))
#
#
#vectorization, num_vocabs = get_text_vectorizer(config_file=None, sequence_length=20, text_data=text_data)
#
#print(num_vocabs)
#
#train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()), vectorization=vectorization, image_size=299, batch_size=4, buffer_size=128)
#valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()), vectorization=vectorization, image_size=299, batch_size=4, buffer_size=128)
#
#for batch_data in train_dataset:
#    batch_img, batch_seq = batch_data
#
#    print(batch_img.numpy().shape, batch_seq.shape)
