from tensorflow.keras import layers, models, Input
from tensorflow.keras.optimizers import Adam

def build_autoencoder():
    input_layer = Input(shape=(128, 128, 128, 1))

    # ----- Encoder -----
    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same', name='enc_conv1')(input_layer)
    x = layers.MaxPooling3D((2, 2, 2), padding='same', name='enc_pool1')(x)

    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='enc_conv2')(x)
    x = layers.MaxPooling3D((2, 2, 2), padding='same', name='enc_pool2')(x)

    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='enc_conv3')(x)
    x = layers.MaxPooling3D((2, 2, 2), padding='same', name='enc_pool3')(x)

    latent = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same', name='latent')(x)

    # ----- Decoder (in autoencoder) -----
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='dec_conv1')(latent)
    x = layers.UpSampling3D((2, 2, 2), name='dec_up1')(x)

    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='dec_conv2')(x)
    x = layers.UpSampling3D((2, 2, 2), name='dec_up2')(x)

    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same', name='dec_conv3')(x)
    x = layers.UpSampling3D((2, 2, 2), name='dec_up3')(x)

    decoded_output = layers.Conv3D(1, (3, 3, 3), activation='linear', padding='same', name='dec_output')(x)

    # Create models
    autoencoder = models.Model(inputs=input_layer, outputs=decoded_output, name='autoencoder')
    encoder = models.Model(inputs=input_layer, outputs=latent, name='encoder')
    
    # Create standalone decoder with SAME layer names
    latent_input = Input(shape=latent.shape[1:], name='decoder_input')
    
    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='dec_conv1')(latent_input)
    x = layers.UpSampling3D((2, 2, 2), name='dec_up1')(x)

    x = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same', name='dec_conv2')(x)
    x = layers.UpSampling3D((2, 2, 2), name='dec_up2')(x)

    x = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same', name='dec_conv3')(x)
    x = layers.UpSampling3D((2, 2, 2), name='dec_up3')(x)

    decoder_output = layers.Conv3D(1, (3, 3, 3), activation='linear', padding='same', name='dec_output')(x)
    
    decoder = models.Model(inputs=latent_input, outputs=decoder_output, name='decoder')

    return autoencoder, encoder, decoder


# Build and train
autoencoder, encoder, decoder = build_autoencoder()
autoencoder.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mse', 'mae']
)

# ... your training code here ...

# After training, save models
autoencoder.save('/data/sai/new_charge/AUGU_20/6059_AE_TRAIN/6059_autoencoder_model_128.h5')
encoder.save('/data/sai/new_charge/AUGU_20/6059_AE_TRAIN/6059_encoder_model_128.h5')

# Transfer trained weights from autoencoder to standalone decoder
for layer in decoder.layers:
    if layer.name in [l.name for l in autoencoder.layers]:
        try:
            ae_layer = autoencoder.get_layer(layer.name)
            if len(ae_layer.get_weights()) > 0:  # Only copy if layer has weights
                layer.set_weights(ae_layer.get_weights())
                print(f"Copied weights for layer: {layer.name}")
        except Exception as e:
            print(f"Could not copy weights for {layer.name}: {e}")

# Save decoder with trained weights
decoder.save('/data/sai/new_charge/AUGU_20/6059_AE_TRAIN/6059_decoder_model_128.h5')