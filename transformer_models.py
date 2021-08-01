from typing import Tuple, List

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def get_cnn_model(image_size: int) -> Tuple[tf.keras.Model, int, int]:
    base_model = efficientnet.EfficientNetB0(input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    base_model_out = base_model.output
    num_features = base_model_out.shape[-1]
    base_model_out = layers.Reshape((-1, num_features))(base_model_out)
    flatten_dim = base_model_out.shape[1]
    cnn_model = keras.models.Model(base_model.input, base_model_out, name="cnn_model")
    return cnn_model, flatten_dim, num_features


def get_angles(pos: np.ndarray, i: np.ndarray, d_model: int) -> float:
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, :, :]

    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


def get_encoder_block(feature_dim: int=1280, embed_dim: int=512,
                      d_ff: int=1024, num_heads: int=8, rate: float=0.4) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, feature_dim), dtype=tf.float32, name="encoder_block_inputs")

    multi_head_attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    #pos_enc = positional_encoding(position=flatten_dim, d_model=feature_dim)

    #mha_input = inputs + pos_enc
    mha_input = inputs

    ffn = point_wise_feed_forward_network(d_model=embed_dim, dff=d_ff)

    layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    dropout1 = tf.keras.layers.Dropout(rate)
    dropout2 = tf.keras.layers.Dropout(rate)

    attention_output, _ = multi_head_attention(query=mha_input, key=mha_input, value=mha_input,
                                               attention_mask=None, return_attention_scores=True)
    attention_output = dropout1(attention_output)

    out_1 = layernorm1(attention_output + mha_input)

    ffn_output = ffn(out_1)
    ffn_output = dropout2(ffn_output)
    out_2 = layernorm2(out_1 + ffn_output)

    return tf.keras.Model(inputs=[inputs], outputs=[out_2])


def get_encoder_model(flatten_dim: int=100, feature_dim: int=1280, embed_dim: int=512,
                      d_ff: int=1024, num_heads: int=8, num_layers: int=6, rate: float=0.4):
    inputs = tf.keras.Input(shape=(None, feature_dim), dtype=tf.float32, name="encoder_inputs")

    pos_encoding = positional_encoding(position=flatten_dim, d_model=embed_dim)

    dropout = tf.keras.layers.Dropout(rate)

    if feature_dim != embed_dim:
        projected_inputs = tf.keras.layers.Dense(embed_dim, activation="relu")(inputs)
    else:
        projected_inputs = inputs

    x = projected_inputs + pos_encoding
    x = dropout(x)


    assert num_layers >= 1
    encoder_layers = [get_encoder_block(feature_dim=embed_dim, embed_dim=embed_dim,
                                        d_ff=d_ff, num_heads=num_heads, rate=rate) for _ in range(num_layers)]

    for i in range(num_layers):
        x = encoder_layers[i](x)

    return tf.keras.Model(inputs=[inputs], outputs=[x], name="encoder")


def get_decoder_block(embed_dim: int=512, d_ff: int=1024, num_heads: int=8, rate: float=0.4) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, embed_dim), dtype=tf.float32, name="decoder_block_inputs")
    encoder_output = tf.keras.Input(shape=(None, embed_dim), dtype=tf.float32)
    look_ahead_mask = tf.keras.Input(shape=(None, None))
    padding_mask = tf.keras.Input(shape=(None, None))

    multi_head_attention_1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
    multi_head_attention_2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)

    ffn = point_wise_feed_forward_network(d_model=embed_dim, dff=d_ff)

    dropout1 = tf.keras.layers.Dropout(rate)
    dropout2 = tf.keras.layers.Dropout(rate)
    dropout3 = tf.keras.layers.Dropout(rate)

    layernorm_1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    layernorm_2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    layernorm_3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    attention_output_1, attention_weight_1 = multi_head_attention_1(query=inputs, key=inputs, value=inputs,
                                                                    attention_mask=look_ahead_mask,
                                                                    return_attention_scores=True)
    attention_output_1 = dropout1(attention_output_1)

    out_1 = layernorm_1(attention_output_1 + inputs)

    attention_output_2, attention_weight_2 = multi_head_attention_2(query=out_1,
                                                                    key=encoder_output, value=encoder_output,
                                                                    attention_mask=padding_mask,
                                                                    return_attention_scores=True)
    attention_output_2 = dropout2(attention_output_2)

    out_2 = layernorm_2(attention_output_2 + out_1)

    ffn_output = ffn(out_2)
    ffn_output = dropout3(ffn_output)

    out_3 = layernorm_3(ffn_output + out_2)

    return tf.keras.Model(inputs=[inputs, encoder_output, look_ahead_mask, padding_mask], outputs=[out_3])


def compute_mask(inputs: tf.Tensor) -> tf.Tensor:
    mask = tf.not_equal(inputs, 0)
    return mask


def get_causal_attention_mask(inputs):
    input_shape = tf.shape(inputs)
    batch_size, sequence_length = input_shape[0], input_shape[1]
    i = tf.range(sequence_length)[:, tf.newaxis]
    j = tf.range(sequence_length)
    mask = tf.cast(i >= j, dtype="int32")
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
        axis=0,
    )
    return tf.tile(mask, mult)


def get_decoder_model(sequence_length: int=25, num_vocabs: int=10000, embed_dim: int=512,
                      d_ff: int=1024, num_heads: int=8, num_layers: int=6, rate: float=0.4) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(sequence_length - 1), dtype=tf.int32, name="decoder_inputs")
    encoder_output = tf.keras.Input(shape=(None, embed_dim), dtype=tf.float32, name="encoder_output")

    mask = compute_mask(inputs=inputs)

    padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
    causal_mask = get_causal_attention_mask(inputs=inputs)
    combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
    combined_mask = tf.minimum(combined_mask, causal_mask)

    embedding = tf.keras.layers.Embedding(num_vocabs, embed_dim)
    pos_encoding = positional_encoding(position=sequence_length - 1, d_model=embed_dim)
    dropout = tf.keras.layers.Dropout(rate)

    assert num_layers >= 1
    decoder_layers = [get_decoder_block(embed_dim=embed_dim, d_ff=d_ff,
                                        num_heads=num_heads, rate=rate) for _ in range(num_layers)]

    dense_out = tf.keras.layers.Dense(num_vocabs)

    x = embedding(inputs) + pos_encoding
    x = dropout(x)

    for i in range(num_layers):
        x = decoder_layers[i]([x, encoder_output, combined_mask, padding_mask])

    out = dense_out(x)

    return tf.keras.Model(inputs=[inputs, encoder_output], outputs=[out], name="decoder")


def get_image_captioning_model(image_size: int=299, embed_dim: int=512,
                               d_ff: int=1024, num_heads: int=8, num_layers: int=6,
                               sequence_length: int=25, num_vocabs: int=10000,
                               rate: float=0.4) -> Tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(image_size, image_size, 3))

    cnn_model, flatten_dim, num_features = get_cnn_model(image_size=image_size)

    out_cnn = cnn_model(inputs)

    #flatten_dim = out_cnn.shape[1]
    #feature_dim = out_cnn.shape[-1]

    encoder_model = get_encoder_model(flatten_dim=flatten_dim, feature_dim=num_features, embed_dim=embed_dim,
                                      d_ff=d_ff, num_heads=num_heads, num_layers=num_layers, rate=rate)

    out_encoder = encoder_model(out_cnn)

    image_and_encoder_model = tf.keras.Model(inputs=[inputs], outputs=[out_encoder])

    decoder_inputs = tf.keras.Input(shape=(sequence_length - 1))

    decoder_model = get_decoder_model(sequence_length=sequence_length, num_vocabs=num_vocabs, embed_dim=embed_dim,
                                      d_ff=d_ff, num_heads=num_heads, num_layers=num_layers, rate=rate)

    out_decoder = decoder_model([decoder_inputs, out_encoder])

    final_model = tf.keras.Model(inputs=[inputs, decoder_inputs], outputs=[out_decoder])

    return image_and_encoder_model, final_model


#image_and_encoder_model, final_model = get_image_captioning_model(num_heads=2, num_layers=1)
#image_and_encoder_model.summary()
#final_model.summary()
#
#final_model.save("final_model")
#
#tf.keras.models.load_model("final_model")

#import tensorflowjs
#
#tensorflowjs.converters.save_keras_model(final_model, "final_model")

#inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name="decoder_inputs")
#mask = compute_mask(inputs=inputs)
#
#padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
#causal_mask = get_causal_attention_mask(inputs=inputs)
#combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
#combined_mask = tf.minimum(combined_mask, causal_mask)
#print(padding_mask.shape, combined_mask.shape)
#decoder_layer = get_decoder_block(look_ahead_mask=combined_mask, padding_mask=padding_mask,
#                                embed_dim=512, d_ff=2048, num_heads=2)


class ImageCaptioningModel(keras.Model):
    def __init__(
        self, cnn_model: tf.keras.Model, encoder: tf.keras.Model, decoder: tf.keras.Model, num_captions_per_image: int=5,
    ):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image

        self.cnn_model.summary()
        self.encoder.summary()
        self.decoder.summary()

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                # 3. Pass image embeddings to encoder
                encoder_out = self.encoder(img_embed, training=True)

                batch_seq_inp = batch_seq[:, i, :-1]
                batch_seq_true = batch_seq[:, i, 1:]

                # 4. Compute the mask for the input sequence
                mask = tf.math.not_equal(batch_seq_inp, 0)

                # 5. Pass the encoder outputs, sequence inputs along with
                # mask to the decoder
                batch_seq_pred = self.decoder([batch_seq_inp, encoder_out], training=True)

                # 6. Calculate loss and accuracy
                caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
                caption_acc = self.calculate_accuracy(
                    batch_seq_true, batch_seq_pred, mask
                )

                # 7. Update the batch loss and batch accuracy
                batch_loss += caption_loss
                batch_acc += caption_acc

            # 8. Get the list of all the trainable weights
            train_vars = (
                self.encoder.trainable_variables + self.decoder.trainable_variables
            )

            # 9. Get the gradients
            grads = tape.gradient(caption_loss, train_vars)

            # 10. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        loss = batch_loss
        acc = batch_acc / float(self.num_captions_per_image)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        img_embed = self.cnn_model(batch_img)

        # 2. Pass each of the five captions one by one to the decoder
        # along with the encoder outputs and compute the loss as well as accuracy
        # for each caption.
        for i in range(self.num_captions_per_image):
            # 3. Pass image embeddings to encoder
            encoder_out = self.encoder(img_embed, training=False)

            batch_seq_inp = batch_seq[:, i, :-1]
            batch_seq_true = batch_seq[:, i, 1:]

            # 4. Compute the mask for the input sequence
            mask = tf.math.not_equal(batch_seq_inp, 0)

            # 5. Pass the encoder outputs, sequence inputs along with
            # mask to the decoder
            batch_seq_pred = self.decoder([batch_seq_inp, encoder_out], training=False)

            # 6. Calculate loss and accuracy
            caption_loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            caption_acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

            # 7. Update the batch loss and batch accuracy
            batch_loss += caption_loss
            batch_acc += caption_acc

        loss = batch_loss
        acc = batch_acc / float(self.num_captions_per_image)

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]
