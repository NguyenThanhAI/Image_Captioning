import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=1000):
        super(LearningRateSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


#temp_learning_rate_schedule = LearningRateSchedule(512)
#plt.plot(temp_learning_rate_schedule(tf.range(400000, dtype=tf.float32)))
#plt.ylabel("Learning Rate")
#plt.xlabel("Train Step")
#plt.show()


class EarlyStoppingAtMaxAccuracy(tf.keras.callbacks.Callback):
    """Stop training when the loss is at its min, i.e. the loss stops decreasing.

  Arguments:
      patience: Number of epochs to wait after min has been hit. After this
      number of no improvement, training stops.
  """

    def __init__(self, monitor="val_acc", patience=0, save_dir="."):
        super(EarlyStoppingAtMaxAccuracy, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = (None, None)

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = - np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if np.greater(current, self.best):
            print("{} improved from {} to {}".format(self.monitor, self.best, current))
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = (self.model.encoder.get_weights(),
                                 self.model.decoder.get_weights())
            self.save_models(epoch=epoch)
        else:
            self.wait += 1
            print("{} does not improve from {}".format(self.monitor, self.best))
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.encoder.set_weights(self.best_weights[0])
                self.model.decoder.set_weights(self.best_weights[1])

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("The {} does not improve from {} for {} epochs. Epoch {}: early stopping".format(self.monitor, self.best, self.patience, self.stopped_epoch + 1))

    def save_models(self, epoch):
        self.model.cnn_model.save(os.path.join(self.save_dir, "cnn_model_epoch_{}_val_acc_{}".format(epoch + 1, self.best)))
        self.model.encoder.save(os.path.join(self.save_dir, "encoder_model_epoch_{}_val_acc_{}".format(epoch + 1, self.best)))
        self.model.decoder.save(os.path.join(self.save_dir, "decoder_model_epoch_{}_val_acc_{}".format(epoch + 1, self.best)))
        print("Save cnn model, encoder, decoder")
