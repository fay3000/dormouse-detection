import datetime
import tensorflow as tf
import numpy as np
import os
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras_cv import visualization

from config import Config
from src.data_loader import DataLoader
from src.model import ObjectDetection
from src.metrics_callback import EvaluateCOCOMetricsCallback
from src.logger import logger


class TrainingPipeline:
    def __init__(self, config: Config):
        """
        Args:
            config (Config): Parameters for training the model.
        """
        self.config = config
        self.bbox_format = config.bbox_format

        # Generate a timestamp for logging/ model naming
        self.log_timestamp = datetime.datetime.now().strftime("%-d-%-m_%H-%M-%S")

        # Initialise saved_model_path as None, to be set when a model is saved
        self.saved_model_path = None


    def load_data(self):
        """
        Loads and prepares data for training and testing.

        Returns:
            train_imgs (ndarray): Preprocessed training images. Shape=(num_images, width, height, channels)
            test_imgs (ndarray): Preprocessed test images. Shape=(num_images, width, height, channels)
            train_labels (dict) : Dictionary of bounding boxes and classes for each training image
                                  in TensorFlow's standard format.
            test_labels (dict): Dictionary of bounding boxes and classes for each test image
                                in TensorFlow's standard format.
        """
        data_loader = DataLoader(self.config)

        (train_images,
         test_images,
         train_labels,
         test_labels) = data_loader.prepare_data(augment=self.config.augmentation)

        logger.info(f"Prepared Datasets. Sizes: Train: {len(train_images)}, "
                    f"Validation: {len(test_images)}")

        return train_images, test_images, train_labels, test_labels


    def train(self, train_images, val_images, train_labels, val_labels)-> None:
        """
           Trains a YOLOV8 object detection model. 
        """
        detector = ObjectDetection()
        model = detector.create_yolov8_model(self.config.num_classes, self.bbox_format)
        logger.info("Initialised Model")

        logger.info("Training Model...")
        self.config.log_training_params()

        model.fit(train_images,
                  train_labels,
                  batch_size=self.config.batch_size,
                  epochs=self.config.epochs,
                  validation_data=(val_images, val_labels),  # For early stopping
                  callbacks=self.create_tensorboard_callbacks(val_images, val_labels))
        logger.info("Finished Training!")

        model_path = f"trained_models/dormouse_model_{self.log_timestamp}.keras"
        model.save(model_path)
        self.saved_model_path = model_path
        logger.info(f"Saved Model: {model_path}")
        return

    def evaluate(self, test_images, test_labels) -> None:
        """
           Visualises the test set results, saving the output to the `plots` dir.
        """
        try:
            if self.saved_model_path:
                logger.info(f'Evaluating model from last train: {self.saved_model_path}')
                model_path = self.saved_model_path
            else:
                logger.info(f'Evaluating model from config saved_model_path: {self.config.saved_model_path}')
                model_path = self.config.saved_model_path

            detector = ObjectDetection()
            model = detector.load_model(model_path)

            model.prediction_decoder.confidence_threshold = 0.2
            # model.prediction_decoder.iou_threshold = 0.7  # default is 0.7, uncomment to change

            logger.info("Predicting Test Images")
            y_pred = model.predict(test_images)

            # Create the output directory if it doesn't exist
            model_name = model_path.split(".")[0].split("/")[-1]
            plots_dir = f"plots/{model_name}"
            os.makedirs(plots_dir, exist_ok=True)

            # Define the path to save the plot
            conf_pct = int(model.prediction_decoder.confidence_threshold * 100)
            path = os.path.join(plots_dir, f"testset_evaluation_confidence_{conf_pct}.png")

            # Plot the predictions
            class_ids = self.config.class_ids
            class_mapping = dict(zip(range(len(class_ids)), class_ids))
            visualization.plot_bounding_box_gallery(
                test_images,
                value_range=(0, 255),
                rows=5, # todo dynamic values
                cols=5,
                y_pred=y_pred,
                y_true=test_labels,
                scale=4,
                font_scale=0.7,
                bounding_box_format=self.bbox_format,
                class_mapping=class_mapping,
                path=path
            )
            logger.info(f"Saved Evaluation Image: {path}")

        except Exception as e:
            logger.warning(f"Failed to evaluate model: {e}")


    def create_tensorboard_callbacks(self, val_images, val_labels) -> list[object]:
        """
        Creates a list of TensorFlow callbacks for model training including TensorBoard 
        logging, COCO metrics evaluation, model checkpoints, and optional early stopping.

        Returns:
              A list of callback objects to be used during model training. 
        """
        log_dir = f"logs/fit/{self.log_timestamp}"

        # Create TensorBoard callback
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Create COCO metrics callback, evaluating the val set after each epoch
        images_tf = tf.convert_to_tensor(np.array(val_images), dtype=tf.float32)
        val_ds = tf.data.Dataset.from_tensor_slices((images_tf, val_labels))
        # create a batched dataset where each element (batch) contains one item from the original dataset
        val_ds_batched = val_ds.batch(1)
        coco_metrics_callback = EvaluateCOCOMetricsCallback(val_ds_batched, self.bbox_format, log_dir)

        # Create callback to save best performing model on the val set
        checkpoint_callback = ModelCheckpoint(filepath=f"trained_models/dormouse_model_{self.log_timestamp}_best.keras",
                                              save_best_only=True)

        callbacks = [tensorboard_callback, coco_metrics_callback, checkpoint_callback]

        # Create early stopping callback
        if self.config.early_stopping:
            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stop_patience, # Number of epochs with no improvement to stop training at
                restore_best_weights=True,  # Restore the weights of the best epoch
                start_from_epoch=0,  # Epoch to start monitoring improvement
                verbose=1,
            )
            callbacks.append(early_stopping_callback)
        return callbacks
