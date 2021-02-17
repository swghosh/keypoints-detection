import numpy as np
import tensorflow as tf


def get_keypoints(kp_file_path):
    """Get keypoints from a .pts file from the 300w dataset.
    
    Args:
        kp_file_path: path to .pts file containing keypoints
    
    Returns:
        an array of shape (68, 2)
    """
    with open(kp_file_path) as f:
        contents = f.read()

    contents = contents.split('{')[1]
    contents = contents.split('}')[0]

    lines = contents.strip().split('\n')
    points = [
        tuple(map(float, kp.split(' ')))
        for kp in lines]

    return np.array(points, dtype=np.float32)


def resize_image_and_adjust_keypoint(image, keypoints, target_height, target_width):
    """Resize an image using padding by maintaing it's aspect
    ratio. The associated keypoints for the image are also adjusted
    accordingly."""
    
    original_size = tf.shape(image)[:-1]  # omit last axis for channels
    resized_image = tf.image.resize_with_pad(
        image, target_height, target_width)
    
    min_dim = tf.argmin(original_size, axis=0)
    min_dim_bool = tf.math.equal(min_dim, 0)  # 0 when height is less than width, 1 otherwise

    if min_dim_bool:
        padding_vec = tf.constant([0, 1], dtype=tf.float32)
        aspect_ratio = tf.cast(original_size[0] / original_size[1], tf.float32)
        shorter_side = aspect_ratio * tf.cast(target_width, tf.float32)
        side_pad = target_height - shorter_side
        fin_size = tf.cast([shorter_side, target_width], tf.float32)
    else:
        padding_vec = tf.constant([1, 0], dtype=tf.float32)
        aspect_ratio = tf.cast(original_size[1] / original_size[0], tf.float32)
        shorter_side = aspect_ratio * tf.cast(target_height, tf.float32)
        side_pad = target_width - shorter_side
        fin_size = tf.cast([target_height, shorter_side], tf.float32)
    padding_vec = padding_vec * (side_pad / 2)

    adjusted_keypoints = keypoints / tf.cast(original_size, tf.float32)
    adjusted_keypoints = adjusted_keypoints * fin_size 
    adjusted_keypoints = adjusted_keypoints + padding_vec
    
    return resized_image, adjusted_keypoints


if __name__ == '__main__':
    import cv2

    image = cv2.imread(
        "../300w_cropped/02_Outdoor/outdoor_300.png")
    kp = get_keypoints(
        "../300w_cropped/02_Outdoor/outdoor_300.pts")

    target_size = 224
    image, kp = resize_image_and_adjust_keypoint(image, kp, target_size, target_size)
    
    kp = kp.numpy().astype(np.int32)
    image = image.numpy().astype(np.uint8)

    for k in kp:
        image = cv2.circle(image, (k[0], k[1]), 2, (0, 0, 255), 2)
    cv2.imwrite('a.png', image)
