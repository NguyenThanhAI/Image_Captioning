from typing import Tuple, List

import tensorflow as tf
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


def get_cnn_model(image_size: int) -> tf.keras.Model:
    base_model = efficientnet.EfficientNetB0(
        input_shape=(image_size, image_size, 3), include_top=False, weights="imagenet",
    )
    base_model.trainable = False
    base_model_out = base_model.output
    num_features = base_model_out.shape[-1]
    base_model_out = tf.keras.layers.Reshape((-1, num_features))(base_model_out)
    cnn_model = tf.keras.models.Model(base_model.input, base_model_out)
    return cnn_model


class RecurrentEncoder(tf.keras.Model):
    def __init__(self, hidden_dim: int=512, dropout_rate: float=0.5, type: str="lstm"):
        super(RecurrentEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.type = type
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        if self.type == "lstm":
            self.recurrent_layer = tf.keras.layers.LSTM(units=self.hidden_dim, return_state=True)
        elif self.type == "gru":
            self.recurrent_layer = tf.keras.layers.GRU(units=self.hidden_dim, return_state=True)
        else:
            raise ValueError("Unknown type {} of recurrent model".format(self.type))

    def call(self, inputs, training=None, mask=None):
        x = self.dropout(inputs, training=training)
        _, state_h, state_c = self.recurrent_layer(x)

        hidden_states = (state_h, state_c)
        
        return hidden_states
    

class RecurrentDecoder(tf.keras.Model):
    def __init__(self, hidden_dim: int=512, dropout_rate: float=0.5, type: str="lstm",
                 embed_dim: int=512, ff_dim: int=2048, vocab_size: int=10000):
        super(RecurrentDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.type = type
        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout_3 = tf.keras.layers.Dropout(rate=dropout_rate)
        if self.type == "lstm":
            self.recurrent_layer = tf.keras.layers.LSTM(units=self.hidden_dim , return_sequences=True)
        elif self.type == "gru":
            self.recurrent_layer = tf.keras.layers.GRU(units=self.hidden_dim, return_sequences=True)
        else:
            raise ValueError("Unknown type {} of recurrent model".format(self.type))
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, mask_zero=True)
        self.dense = tf.keras.layers.Dense(units=ff_dim, activation="relu")
        self.out = tf.keras.layers.Dense(units=vocab_size, activation=None)

    def call(self, inputs, hidden_states, training=None, mask=None):
        x = self.embedding_layer(inputs)
        x = self.dropout_1(x, training=training)
        x = self.recurrent_layer(x, initial_state=hidden_states, mask=mask)
        x = self.dropout_2(x, training=training)
        x = self.dense(x)
        x = self.dropout_3(x, training=training)
        x = self.out(x)

        return x


class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_model: tf.keras.Model, encoder: RecurrentEncoder,
                 decoder: RecurrentDecoder, num_captions_per_image: int=5):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")
        self.num_captions_per_image = num_captions_per_image


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

    def _compute_loss_and_acc(self, batch_data, training=True):
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
                encoder_out = self.encoder(inputs=img_embed, training=training)

                batch_seq_inp = batch_seq[:, i, :-1]
                batch_seq_true = batch_seq[:, i, 1:]

                # 4. Compute the mask for the input sequence
                mask = tf.math.not_equal(batch_seq_inp, 0)

                # 5. Pass the encoder outputs, sequence inputs along with
                # mask to the decoder
                batch_seq_pred = self.decoder(inputs=batch_seq_inp, hidden_states=encoder_out, training=training, mask=mask)

                # 6. Calculate loss and accuracy
                loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
                acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)

                # 7. Update the batch loss and batch accuracy
                batch_loss += loss
                batch_acc += acc

            # 8. Get the list of all the trainable weights
            train_vars = (self.encoder.trainable_variables + self.decoder.trainable_variables)

            # 9. Get the gradients
            grads = tape.gradient(loss, train_vars)

            # 10. Update the trainable weights
            self.optimizer.apply_gradients(zip(grads, train_vars))

        return batch_loss, batch_acc / float(self.num_captions_per_image)

    def train_step(self, batch_data):
        loss, acc = self._compute_loss_and_acc(batch_data)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    def test_step(self, batch_data):
        loss, acc = self._compute_loss_and_acc(batch_data, training=False)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]
