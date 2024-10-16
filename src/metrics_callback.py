import keras_cv
import keras
import tensorflow as tf


class EvaluateCOCOMetricsCallback(keras.callbacks.Callback):
    def __init__(self, data, bounding_box_format, log_dir):
        super().__init__()
        self.data = data
        self.metrics = keras_cv.metrics.BoxCOCOMetrics(
            bounding_box_format=bounding_box_format,
            evaluate_freq=1e9,  # Disables auto evaluation; metrics computed only in on_epoch_end()
        )
        self.best_map = -1.0
        self.log_dir = log_dir
        self.tensorboard_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs):
        self.metrics.reset_state()
        for batch in self.data:
            images, y_true = batch[0], batch[1]
            y_pred = self.model.predict(images, verbose=0)
            self.metrics.update_state(y_true, y_pred)

        metrics = self.metrics.result(force=True)
        logs.update(metrics)

        # Log the COCO metrics to TensorBoard
        with self.tensorboard_writer.as_default():
            for metric_name, metric_value in metrics.items():
                tf.summary.scalar(metric_name, metric_value, step=epoch)

        return logs
