

import tensorflow as tf # TF 2.2.0

# Download MobileNetv2 (using tf.keras)
# BAD: keras_model = tf.keras.applications.MobileNetV2(
# GOOD: keras_model = tf.keras.applications.MobileNet(
# GOOD: keras_model = tf.keras.applications.MobileNetV3Small(   # size 48
keras_model = tf.keras.applications.MobileNetV3Large(
    weights="imagenet", 
    input_shape=(224, 224, 3,),
    classes=1000,
)

from sys import getsizeof
print(getsizeof(keras_model))

print('all done')