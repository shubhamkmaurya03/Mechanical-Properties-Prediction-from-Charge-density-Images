import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import GlobalAveragePooling3D, GlobalMaxPooling3D, Dense, Reshape, Multiply, Add, Conv3D

# Note: The @register_keras_serializable decorator allows the .h5/.keras file
# to be loaded successfully, as it links the saved model's config
# back to these Python classes.

@tf.keras.utils.register_keras_serializable()
class DualAttention3D(layers.Layer):
    """
    Custom Dual-Attention (Channel + Spatial) Layer.
    Required to load the saved Keras model.
    """
    def __init__(self, reduction_ratio=16, **kwargs):
        super(DualAttention3D, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.global_avg_pool=GlobalAveragePooling3D()
        self.global_max_pool=GlobalMaxPooling3D()
        self.dense1=Dense(units=max(self.channels // self.reduction_ratio, 8), activation='relu')
        self.dense2=Dense(units=self.channels, activation='sigmoid')
        self.conv1=Conv3D(2, 3, padding='same', activation='relu')
        self.conv2=Conv3D(1, 3, padding='same', activation='sigmoid')
        super(DualAttention3D, self).build(input_shape)

    def call(self, inputs):
        avg_pool=self.global_avg_pool(inputs); max_pool=self.global_max_pool(inputs)
        avg_pool=Reshape((1,1,1,self.channels))(avg_pool); max_pool=Reshape((1,1,1,self.channels))(max_pool)
        channel_avg=self.dense2(self.dense1(avg_pool)); channel_max=self.dense2(self.dense1(max_pool)); channel_attention=Add()([channel_avg, channel_max])
        x=Multiply()([inputs, channel_attention]); spatial_avg=tf.reduce_mean(x, axis=-1, keepdims=True); spatial_max=tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_concat=tf.concat([spatial_avg, spatial_max], axis=-1); spatial_attention=self.conv2(self.conv1(spatial_concat))
        output=Multiply()([x, spatial_attention]); return output
    
    def get_config(self):
        config = super(DualAttention3D, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config

@tf.keras.utils.register_keras_serializable()
class RandomFlip3D(layers.Layer):
    """
    Custom Random Flip 3D Layer (using tf.cond).
    Required to load the saved Keras model.
    """
    def __init__(self, seed=None, **kwargs):
        super(RandomFlip3D, self).__init__(**kwargs)
        self.seed = seed

    def call(self, inputs, training=None):
        # During prediction (training=False), this layer does nothing.
        if training:
            x = inputs
            x = tf.cond(tf.random.uniform((), seed=self.seed) > 0.5, lambda: tf.reverse(x, axis=[1]), lambda: x)
            x = tf.cond(tf.random.uniform((), seed=self.seed) > 0.5, lambda: tf.reverse(x, axis=[2]), lambda: x)
            x = tf.cond(tf.random.uniform((), seed=self.seed) > 0.5, lambda: tf.reverse(x, axis=[3]), lambda: x)
            return x
        return inputs
        
    def get_config(self):
        config = super(RandomFlip3D, self).get_config()
        config.update({"seed": self.seed})
        return config