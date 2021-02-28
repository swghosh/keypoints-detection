"""Train a Keras model to detect keypoints."""
import tensorflow as tf
from keypoints_detection import dataset


image_size = 224
num_keypoints = 68
keypoints_coords = 2
dataset_path = '../300w_cropped'
batch_size = 32
shuffle_buffer = 16
epochs = 50
train_split = 0.9
num_samples = 600
shuffle_seed = 91
cross_val_num_folds = 10


def create_model(finetune=False):
    """Creates a ResNet50V2 model for keypoints detection."""

    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet',
        input_shape=(image_size, image_size, 3),
        pooling='avg')
    base_model.trainable = finetune

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Dense(num_keypoints * keypoints_coords),
        tf.keras.layers.Reshape([num_keypoints, keypoints_coords])
    ])
    model.compile('rmsprop', 'mse')

    return model


def prepare_and_get_dataset():
    """Construct `tf.data.Dataset` pipeline(s) for training and
    validation splits and return them."""

    ds = tf.data.Dataset.from_generator(
        lambda: dataset.load_dataset_as_generator(
            dataset_path, image_size),
        output_signature=(tf.TensorSpec(shape=(image_size,
                                               image_size,
                                               3),
                                        dtype=tf.uint8),
                          tf.TensorSpec(shape=(num_keypoints,
                                               keypoints_coords),
                                        dtype=tf.float32)))

    def preprocess_samples(image, keypoints):
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.resnet_v2.preprocess_input(image)
        # normalize keypoints scaling it by image size
        keypoints = keypoints / image_size
        return image, keypoints

    ds = ds.shuffle(shuffle_buffer,
                    seed=shuffle_seed,
                    reshuffle_each_iteration=False)
    ds = ds.map(preprocess_samples, tf.data.experimental.AUTOTUNE)

    train_samples = round(num_samples * train_split)
    train_ds = ds.take(train_samples).cache()
    val_ds = ds.skip(train_samples).cache()

    # additionally, shuffle train samples
    train_ds = train_ds.shuffle(
        shuffle_buffer,
        reshuffle_each_iteration=True).batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    return train_ds, val_ds


for _ in range(cross_val_num_folds):
    tf.keras.backend.clear_session()
    model = create_model()
    train_ds, val_ds = prepare_and_get_dataset()
    model.fit(train_ds,
              epochs=epochs,
              validation_data=val_ds)
    shuffle_seed += 1
