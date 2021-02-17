import tensorflow as tf
from keypoints_detection import dataset


image_size = 224
num_keypoints = 68
keypoints_coords = 2
dataset_path = '../300w_cropped'
batch_size = 32
shuffle_buffer = 16
epochs = 50
validation_split = 0.3

base_model = tf.keras.applications.ResNet50V2(include_top=False,
                                              weights='imagenet',
                                              input_shape=(image_size, image_size, 3),
                                              pooling='avg')
base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dense(num_keypoints * keypoints_coords),
    tf.keras.layers.Reshape([num_keypoints, keypoints_coords])
])
model.compile('rmsprop', 'mse')

dataset_gen = dataset.load_dataset_as_generator(
    dataset_path, image_size)
ds = tf.data.Dataset.from_generator(lambda: dataset_gen,
                                    output_signature=(
                                        tf.TensorSpec(shape=(image_size, image_size, 3),
                                                      dtype=tf.uint8),
                                        tf.TensorSpec(shape=(num_keypoints, keypoints_coords),
                                                      dtype=tf.float32)))
ds = ds.cache()


def preprocess_samples(image, keypoints):
    image = tf.cast(image, tf.float32)
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    keypoints = keypoints / image_size  # normalize keypoints scaling it by image size
    return image, keypoints

ds = ds.shuffle(shuffle_buffer)
ds = ds.map(preprocess_samples, tf.data.experimental.AUTOTUNE)
ds = ds.batch(batch_size)

model.fit(ds, epochs=epochs)
