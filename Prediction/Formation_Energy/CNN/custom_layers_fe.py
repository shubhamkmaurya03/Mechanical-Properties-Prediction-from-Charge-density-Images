import tensorflow as tf
from tensorflow.keras import layers, models

@tf.keras.utils.register_keras_serializable()
class DualAttention3D(layers.Layer):
    """ Custom Dual-Attention (Channel + Spatial) Layer """
    def __init__(self, reduction_ratio=16, **kwargs):
        super(DualAttention3D, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.global_avg_pool = layers.GlobalAveragePooling3D()
        self.global_max_pool = layers.GlobalMaxPooling3D()
        self.dense1 = layers.Dense(units=max(self.channels // self.reduction_ratio, 8), activation='relu')
        self.dense2 = layers.Dense(units=self.channels, activation='sigmoid')
        self.conv1 = layers.Conv3D(2, 3, padding='same', activation='relu')
        self.conv2 = layers.Conv3D(1, 3, padding='same', activation='sigmoid')
        super(DualAttention3D, self).build(input_shape)

    def call(self, inputs):
        avg_pool = self.global_avg_pool(inputs)
        max_pool = self.global_max_pool(inputs)
        avg_pool = layers.Reshape((1, 1, 1, self.channels))(avg_pool)
        max_pool = layers.Reshape((1, 1, 1, self.channels))(max_pool)
        
        channel_avg = self.dense2(self.dense1(avg_pool))
        channel_max = self.dense2(self.dense1(max_pool))
        channel_attention = layers.Add()([channel_avg, channel_max])
        
        x = layers.Multiply()([inputs, channel_attention])
        
        spatial_avg = tf.reduce_mean(x, axis=-1, keepdims=True)
        spatial_max = tf.reduce_max(x, axis=-1, keepdims=True)
        spatial_concat = tf.concat([spatial_avg, spatial_max], axis=-1)
        spatial_attention = self.conv2(self.conv1(spatial_concat))
        
        output = layers.Multiply()([x, spatial_attention])
        return output

    def get_config(self):
        config = super(DualAttention3D, self).get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config