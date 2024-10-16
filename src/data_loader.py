""" Class for loading and preparing data for training. """
import os
import keras
import keras_cv
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config import Config
from src.logger import logger
from src.utils import preprocess_image


class DataLoader:
    
    def __init__(self, config: Config):
        """
        Args:
            config (Config): Configuration parameters for training the model.
        """
        self.data_dir = config.dataset_dir
        self.input_shape = config.input_shape
        self.bbox_format = config.bbox_format

    
    def prepare_data(self, augment=True):
        """
        Loads and prepares data for training and testing.

        Args:
            augment (bool, optional): Apply data augmentation to the training set. True by default.
        Returns:
            train_imgs (ndarray): Preprocessed training images. Shape=(num_images, width, height, channels)
            test_imgs (ndarray): Preprocessed test images. Shape=(num_images, width, height, channels)
            train_labels (dict) : Dictionary of bounding boxes and classes for each training image
                                  in TensorFlow's standard format.
            test_labels (dict): Dictionary of bounding boxes and classes for each test image
                                in TensorFlow's standard format.
            For example, for 2 images, each with a single bounding box and class label 0:
            labels = {
                "boxes": tf.constant([
                    [
                        [0, 0, 100, 100]  # image1 bbox
                    ],
                    [
                        [100, 100, 200, 200] # image2 bbox
                    ]
                ], dtype=tf.float32),
                "classes": tf.constant([[0], [0]], dtype=tf.int64)} # bbox class=0 for image 1 & 2
        """
        # Get training and test data filepaths
        image_paths, annotation_paths = self.get_data_filepaths()
        train_paths, test_paths = self.split_dataset(image_paths, annotation_paths)

        # Prepare training and test data
        train_imgs, train_labels = self.preprocess_images_and_annotations(train_paths['image_paths'],
                                                                          train_paths['annotation_paths'])
        test_imgs, test_labels = self.preprocess_images_and_annotations(test_paths['image_paths'],
                                                                        test_paths['annotation_paths'])
        if augment:
            # Create augmented training data
            augmented_train_imgs, augmented_train_labels = self.augment_dataset(train_imgs, train_labels)
            return augmented_train_imgs, test_imgs, augmented_train_labels, test_labels
        else:
            return train_imgs, test_imgs, train_labels, test_labels

    
    def augment_dataset(self, train_images, train_labels, repeat_by=5):
        """
        Augments the given training data by applying image transformations
        such as flipping and jittered resize. Useful for increasing
        training dataset size and variation.

        Args:
        train_images (np.ndarray): Training images.
        train_labels (dict): A dictionary containing label data in TensorFlow's standard format. This
                             should include:
                             - "boxes": Bounding box coordinates for each image.
                             - "classes": Class labels for each bounding box.
        repeat_by (int, optional): The number of times to repeat the dataset before applying augmentation.
                                   Default is 5.

       Returns:
            - np.ndarray: Augmented images.
            - dict: Augmented labels in TensorFlow labels dictionary format.

        """
        # Convert images and labels to TensorFlow dataset
        images_tf = tf.convert_to_tensor(np.array(train_images), dtype=tf.float32)
        train_dataset = tf.data.Dataset.from_tensor_slices((images_tf, train_labels))

        # Repeat the dataset to increase its size
        train_ds = train_dataset.repeat(count=repeat_by)
        # Apply augmentation on the repeated dataset
        augmented_ds = train_ds.map(self.augment, num_parallel_calls=tf.data.AUTOTUNE)

        # Reorganize dataset into a NumPy array of images and a labels dictionary.
        augmented_imgs_list = []
        boxes_list = []
        classes_list = []
        for img, labels in augmented_ds:
            # Convert tensors to NumPy arrays and add to lists
            augmented_imgs_list.append(img.numpy())
            boxes_list.append(labels["boxes"].numpy())
            classes_list.append(labels["classes"].numpy())

        augmented_train_labels = {
            "boxes": tf.convert_to_tensor(np.array(boxes_list), dtype=tf.float32), # todo - sometimes get an error here sometimes dont, must be due to randomness of augmentation, find out the problem: ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2388,) + inhomogeneous part.
            "classes": tf.convert_to_tensor(np.array(classes_list), dtype=tf.int64)
        }
        augmented_train_imgs = np.array(augmented_imgs_list)

        return augmented_train_imgs, augmented_train_labels

    
    @tf.function
    def augment(self, image, labels):
        augmenter = keras.Sequential(
            layers=[
                keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format=self.bbox_format),
                keras_cv.layers.JitteredResize(
                    target_size=(self.input_shape[0], self.input_shape[1]), scale_factor=(0.75, 1.3),
                    bounding_box_format=self.bbox_format,
                ),
            ]
        )
        inputs = {"images": image, "bounding_boxes": labels}
        augmented = augmenter(inputs, training=True)
        return augmented["images"], augmented["bounding_boxes"]


    def preprocess_images_and_annotations(self, image_paths, annotation_paths):
        """
        Resizes and pads the given images to fit `self.input_shape` and adjusts
        their bounding boxes accordingly.

        Args:
            image_paths (list[str]): List of image filepaths.
            annotation_paths (list[str]): List of corresponding annotation filepaths.
        Returns:
            images (np.ndarray): 2D array of preprocessed images. Each image is
                             represented as a 3D array (height, width, channels),
                             resulting in a shape of (num_images, height, width, channels).
            labels_dict (dict): Corresponding image annotations in Tensorflow labels format.
        """
        images = []
        annotations = []
        for img_path, annotation_path in zip(image_paths, annotation_paths):
            img_annot = self.parse_annotations(annotation_path)
            resized_image, resized_annot = self.preprocess_single_image_and_annotation(img_path, img_annot)

            # Exclude images with none or more than one bbox for now.
            # A tf Ragged Dataset is needed to hold varying numbers of bboxes,
            # but it cannot be used as doesn't support training with centre_xywh format, only xyxy.
            if len(resized_annot) == 1:
                images.append(resized_image)
                annotations.append(resized_annot[0])  # Extract the single inner array

        labels_dict = self.format_tensorflow_labels(annotations)

        return np.array(images), labels_dict

    
    def preprocess_single_image_and_annotation(self, image_path, annotations):
        """
        Resizes and pads a single image to the model input_shape, maintaining the
        aspect ratio. Adjusts the bounding box annotations accordingly.

        Args:
            image_path (str): Image filepath.
            annotations (list): List of bounding boxes for the image where the bbox format
                                is relative centre_xywh, range [0-1].
        Returns:
            image (np.ndarray): Resized and padded image.
            adjusted_annotations (list): Adjusted bounding boxes for the image.
        """
        image_padded, new_width, new_height, x_offset, y_offset = preprocess_image(image_path, self.input_shape)

        # Adjust the old bounding boxes to match the resized image
        adjusted_annotations = []
        for annot in annotations:
            adjusted_annot = self.adjust_bbox_for_resized_image(annot, new_width, new_height, x_offset, y_offset)
            adjusted_annotations.append(adjusted_annot)

        return image_padded, adjusted_annotations


    def adjust_bbox_for_resized_image(self, annotation, img_width, img_height, left_offset, top_offset):
        """
        Adjusts a bounding box annotation for an image that has been resized and potentially padded.

        The annotation in the format of normalised bounding box coordinates (relative to the original
        image dimensions) is adjusted and converted to absolute pixel coordinates,
        taking into account the resized image dimensions and any offsets (e.g. padding) applied to the image.

        Params:
            annotation (list or tuple): The bounding box annotation in the format:
                            [class_id, x_centre_relative, y_centre_relative, width_relative, height_relative]
            img_width (int or float): Width of the resized image.
            img_height (int or float): Height of the resized image.
            left_offset (int or float): The horizontal offset (in pixels) applied to the
                                        resized image (e.g., if the image was padded).
            top_offset (int or float) The vertical offset (in pixels) applied to the
                                       resized image (e.g., if the image was padded).

        Returns:
            list: The adjusted bounding box annotation in the format:
                [class_id, x_centre_absolute, y_centre_absolute, width_absolute, height_absolute]
        """
        class_id, x_centre, y_centre, relative_box_w, relative_box_h = annotation
        x_centre = (x_centre * img_width + left_offset)
        y_centre = (y_centre * img_height + top_offset)
        width = (relative_box_w * img_width)
        height = (relative_box_h * img_height)
        return [class_id, x_centre, y_centre, width, height]


    @staticmethod
    def split_dataset(image_paths, annotation_paths, test_size=0.1):
        (imgs_train,
         imgs_test,
         annots_train,
         annots_test) = train_test_split(image_paths,
                                         annotation_paths,
                                         test_size=test_size,
                                         random_state=1,
                                         shuffle=True,
                                         stratify=None)  # todo stratify

        train_paths = {'image_paths': imgs_train, 'annotation_paths': annots_train}
        test_paths = {'image_paths': imgs_test, 'annotation_paths': annots_test}

        logger.info(f"Test image paths: {test_paths['image_paths']}")
        return train_paths, test_paths


    def get_data_filepaths(self):
        """
        Retrieves the filepaths of all training images and
        corresponding annotations in `self.data_dir`. Images are .jpg format and
        annotations are YOLO formatted .txt files with the same filename as the image.

        Returns:
            image_paths (list): List of paths to the .jpg image files.
            annotation_paths (list): List of paths to the corresponding annotation .txt files.
        """
        image_paths = []
        annotation_paths = []

        # Walk through the directory tree, finding data in subdirectories
        for root_dir, dirs, files in os.walk(self.data_dir):
            for filename in files:
                if filename.endswith(".jpg"):
                    image_path = os.path.join(root_dir, filename)
                    annotation_filename = filename.replace(".jpg", ".txt")
                    annotation_path = os.path.join(root_dir, annotation_filename)

                    # Check if the annotation file exists
                    if os.path.isfile(annotation_path):
                        annotation_paths.append(annotation_path)
                        image_paths.append(image_path)
                    else:
                        logger.warning(f"Annotation file {annotation_path} not found for {image_path}.")

        return image_paths, annotation_paths

    
    @staticmethod
    def parse_annotations(annotation_path: str) -> list[list]:
        """
        Reads and parses a text file containing YOLO formatted bounding box
        annotations for a single image.

        The expected format of each line in the file is:
             class_id x_centre y_centre width height

            - class_id: (int) Class ID of the object.
            - x_centre: (float) Normalised x-coordinate of the centre of the bounding box (relative to image width).
            - y_centre: (float) Normalised y-coordinate of the centre of the bounding box (relative to image height).
            - width: (float) Normalised width of the bounding box (relative to image width).
            - height: (float) Normalised height of the bounding box (relative to image height).
        Args:
            annotation_path (str): Annotation filepath.
        Returns:
            parsed_annotations (list[list]): Bounding box annotations for a single image. Each sub list represents
                                             a single bbox annotation.
        """
        parsed_annotations = []
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                annot = line.strip().split(' ')
                class_id = int(annot[0])
                x_centre, y_centre, width, height = map(float, annot[1:])
                parsed_annotations.append([class_id, x_centre, y_centre, width, height])

        return parsed_annotations

    
    @staticmethod
    def format_tensorflow_labels(annotations: list[list]) -> dict:
        """
        Organises annotations into the dictionary format suitable for TensorFlow models.

        Args:
        annotations (list[list]): List of image annotations. Each inner element is a single annotation:
                            [class_id  centre_x  centre_y  width  height]. Note: Zero bounding box
                            (empty list) or multiple bounding boxes are not expected.
                            Shape=(num_images, 5), where 5 is the length of an annotation.
        Returns:
        dict: TensorFlow labels dictionary with keys:
            - "boxes": A tensor of shape (num_images, 1, 4) containing bounding box coordinates.
            - "classes": A tensor of shape (num_images, 1) containing class IDs.
        """
        boxes = []
        classes = []

        # Loop through each annotation and extract the class and box
        for annot in annotations:
            class_id = int(annot[0])
            bbox = annot[1:]
            classes.append([class_id])
            boxes.append([bbox])

        labels_dict = {"boxes": tf.convert_to_tensor(boxes, dtype=tf.float32),
                       "classes": tf.convert_to_tensor(classes, dtype=tf.int64)}
        return labels_dict
