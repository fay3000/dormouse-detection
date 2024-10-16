""" Object detection model architecture and initialisation """
from keras_cv.src.models import YOLOV8Detector
from tensorflow import keras
import keras_cv
from keras.models import load_model as load


class ObjectDetection:
    def __init__(self):
        return

    def create_yolov8_model(self, num_classes: int, bounding_box_format: str) -> YOLOV8Detector:
        """
            Initialises a keras YOLOV8Detector object detection model. The architecture
            uses transfer learning from a pretrained yolov8 model with ResNet50 backbone on
            the imagenet dataset. It requires an input image size that is divisible by 64 for
            the convolution layer operations.

            Returns:
                A pretrained YOLOV8Detector model.
        """

        # Use a pretrained ResNet50 backbone from the imagenet dataset.
        model = keras_cv.models.YOLOV8Detector.from_preset(
            "resnet50_imagenet",
            bounding_box_format=bounding_box_format,
            num_classes=num_classes,
        )
        self.compile(model)
        return model

    def compile(self, model):
        model.compile(
            classification_loss="binary_crossentropy",
            box_loss="ciou",
            optimizer=self.get_optimizer(),
        )

    def get_optimizer(self):
        return keras.optimizers.SGD(
            learning_rate= 0.009, # base learning rate
            momentum=0.9,
            global_clipnorm=10.0
        )

    def load_model(self, model_path: str):
        """
          Loads and compiles a pre-trained model from the specified path.
        """
        model = load(model_path)
        self.compile(model)
        return model

