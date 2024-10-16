import cv2
import numpy as np

def preprocess_image(image_path, input_shape):
    """
    Preprocesses an image for input into a neural network. This involves resizing it to the
    specified input_shape. It maintains the aspect ratio of the original image and adds grey
    coloured padding to the surrounding space.

    Args:
        image_path (str): Input image path
        input_shape (tuple or list): Desired (height, width) or (height, width, channels) of the output image.
    """
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise ValueError(f"Image not found or could not be loaded: {image_path}")

    # Convert the BGR image to RGB format
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Get the current image height and width
    h, w, _ = image.shape

    # Determine the scale factor to resize the image whilst maintaining the aspect ratio
    scale = min(input_shape[1] / w, input_shape[0] / h)
    new_width, new_height = int(w * scale), int(h * scale)

    # Resize the image to the new dimensions
    image_resized = cv2.resize(image, (new_width, new_height))

    padding_colour = 128  # neutral grey
    image_padded = np.full((input_shape[0], input_shape[1], 3), padding_colour)

    # Calculate the offsets needed to place the image in the centre of image_padded
    x_offset = (input_shape[1] - new_width) // 2  # left-side offset
    y_offset = (input_shape[0] - new_height) // 2  # top offset

    # Place the resized image in the centre of the padded image
    image_padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width, :] = image_resized
    return image_padded, new_width, new_height, x_offset, y_offset
