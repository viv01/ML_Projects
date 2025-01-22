import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
from tqdm import tqdm
import tensorflow as tf


def preprocess_audio(file_path):
    """
    Preprocess an audio file into a Mel spectrogram.
    """
    y, sr = librosa.load(file_path, sr=44100)

    # Create Mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db


def prepare_sequences(spectrogram, sequence_length):
    """
    Prepare overlapping sequences for input and output.
    """
    X, y = [], []
    for i in range(len(spectrogram) - sequence_length):
        X.append(spectrogram[i:i + sequence_length])
        y.append(spectrogram[i + sequence_length])
    return np.array(X), np.array(y)

def create_lstm_model(input_shape):
    """
    Create an LSTM-based model for generating variations of a song.
    """
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=input_shape),
        #tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(128, return_sequences=False),
        #tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(input_shape[1], activation='linear')  # Linear activation for regression
    ])
    '''
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(256, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(input_shape[1], activation='linear')
    ])

    #model.compile(optimizer='adam', loss='mse')  # Mean squared error for regression

    ##OPTION 1
    #opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #model.compile(optimizer=opt, loss='mse')

    #OPTION 2
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='mae')

    #OPTION 3
    # opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # def custom_loss(y_true, y_pred):
    #     return tf.reduce_mean(tf.square(tf.math.log1p(y_true - y_pred)))
    # model.compile(optimizer=opt, loss=custom_loss)


    return model


def generate_audio_variation_array(model, spectrogram, sequence_length, batch_size=32):
    """
    Generate a variation of the input spectrogram.
    """
    generated = []
    current_sequence = spectrogram[:sequence_length]
    generated.extend(current_sequence)

    num_frames = len(spectrogram) - sequence_length

    for i in tqdm(range(num_frames), desc="Generating audio variation"):
        input_sequence = np.expand_dims(current_sequence, axis=0)  # Add batch dimension
        input_sequence = np.expand_dims(input_sequence, axis=-1)  # Add channel dimension

        next_frame = model.predict(input_sequence, verbose=0)  # Predict next frame
        generated.append(next_frame[0])  # Append predicted frame

        # Update the current sequence
        current_sequence = np.vstack([current_sequence[1:], next_frame])

    return np.array(generated)


def generate_wav_from_spectogram(file_path, generated_spectrogram, sr):
    
    y, sr = librosa.load(file_path, sr=44100)

    # Convert the decibel spectrogram back to power
    print("generate_wav_from_spectogram() : Convert the decibel spectrogram back to power")
    spectrogram_power = librosa.db_to_power(generated_spectrogram, ref=1.0)

    # Convert power Mel spectrogram to linear-frequency spectrogram
    print("generate_wav_from_spectogram() : Convert power Mel spectrogram to linear-frequency spectrogram")
    mel_basis = librosa.filters.mel(sr=44100, n_fft=2048, n_mels=128)
    S_linear = np.dot(np.linalg.pinv(mel_basis), spectrogram_power)
    S_linear = np.maximum(S_linear, 0)






    '''
    #### with progess bar
    # Ensure S_linear has the correct shape
    n_fft = 1024  # Define the FFT size used in creating S_linear
    S_linear = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512, win_length=n_fft))  # Recompute S_linear
    print(f"S_linear shape: {S_linear.shape}")

    # Reconstruct audio using Griffin-Lim
    print("generate_wav_from_spectogram() : Reconstruct audio using Griffin-Lim")
    #generated_audio = librosa.griffinlim(S_linear, hop_length=512, n_iter=64)
    generated_audio = griffinlim_with_progress(S_linear, n_iter=64, hop_length=512, n_fft=n_fft)
    '''

    #### without progress bar
    # Reconstruct audio using Griffin-Lim
    print("generate_wav_from_spectogram() : Reconstruct audio using Griffin-Lim")
    generated_audio = librosa.griffinlim(S_linear, hop_length=512, n_iter=64)





    # Normalize the audio to [-1, 1]
    print("generate_wav_from_spectogram() : Normalize the audio to [-1, 1]")
    generated_audio = generated_audio / np.max(np.abs(generated_audio))

    # Apply pre-emphasis filter
    print("generate_wav_from_spectogram() : Apply pre-emphasis filter")
    def pre_emphasis(signal, coeff=0.97):
        return np.append(signal[0], signal[1:] - coeff * signal[:-1])

    generated_audio = pre_emphasis(generated_audio)

    # Save the reconstructed audio
    print("generate_wav_from_spectogram() : Save the reconstructed audio")
    sf.write("9a_generated_variation.wav", generated_audio, sr)

    # Visualize the reconstructed spectrogram
    print("generate_wav_from_spectogram() : Visualize the reconstructed spectrogram")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S_linear, ref=np.max), sr=sr, hop_length=512, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Reconstructed Spectrogram")
    plt.show()

def griffinlim_with_progress(S, n_iter=32, hop_length=512, win_length=None, n_fft=None, window="hann", center=True, dtype=None):
    """
    Custom Griffin-Lim algorithm with progress bar.
    Parameters match librosa.griffinlim, with an added progress bar.
    """
    if win_length is None:
        win_length = S.shape[0]
    if n_fft is None:
        n_fft = (S.shape[0] - 1) * 2  # Calculate n_fft from the shape of S

    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    rebuilt = S * angles
    for _ in tqdm(range(n_iter), desc="Griffin-Lim iterations"):
        inverse = librosa.istft(rebuilt, hop_length=hop_length, win_length=win_length, window=window, center=center)
        rebuilt = librosa.stft(inverse, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
        rebuilt = S * np.exp(1j * np.angle(rebuilt))

    inverse = librosa.istft(rebuilt, hop_length=hop_length, win_length=win_length, window=window, center=center)
    return inverse


# --- Main Program ---
if __name__ == "__main__":

    ##########################################################
    # Load and preprocess the audio file
    file_path = r"C:\Users\DELL\Documents\PROJECTS\AI_generated_music\wav_files\Eduardo_Rayez_Jeanway_-_Where_I_W_(getmp3.pro).wav"  # Replace with the path to your audio file
    


    ##########################################################
    spectrogram = preprocess_audio(file_path)
    '''
    print("preprocess_audio : spectogram_db type-> ", type(spectrogram))
    print(np.sort(spectrogram))
    print(len(spectrogram))
    print(len(spectrogram[50]))
    print("*****************************")
    '''
    
    
    ##########################################################

    sequence_length = 86  # Number of time steps in each sequence

    # Prepare sequences
    X, y = prepare_sequences(spectrogram, sequence_length)
    print("preprocess_audio : spectogram_db type-> ", type(spectrogram))
    '''
    print(X.shape)
    print(X.shape[2])
    print(np.sort(X))
    print(len(X))
    print(len(X[41]))
    print(len(X[41][0]))
    print("*****************************")
    '''
    print(y.shape)
    print(np.sort(y))
    print(len(y))
    print(len(y[41]))
    print("*****************************")

    '''
    X = X[..., np.newaxis]  # Add channel dimension for the model
    y = y[..., np.newaxis]  # Target also needs channel dimension
    print(X.shape)
    #print(np.sort(X))
    print(len(X))
    print(len(X[41]))
    print(len(X[41][0]))
    print("*****************************")
    print(y.shape)
    #print(np.sort(y))
    print(len(y))
    print(len(y[41]))
    print("*****************************")
    '''

    ##########################################################

    # Create and train the model
    model = create_lstm_model(input_shape=(sequence_length, X.shape[2]))
    model.summary()


    ##########################################################

    #model.fit(X, y, epochs=15, batch_size=64)
    model.fit(X, y, epochs=15, batch_size=16)



    ##########################################################

    # Save the trained model
    model.save("9a_song_variation_model.h5")


    
    ##########################################################

    # Generate a variation of the song
    generated_spectrogram = generate_audio_variation_array(model, spectrogram, sequence_length)
    print(generated_spectrogram.shape)
    print(np.sort(generated_spectrogram))
    print(len(X))
    print("*****************************")

    '''
    #generated_spectrogram =  spectrogram
    

    ##########################################################

    # Convert the generated spectrogram back to audio and show plotted graph
    generate_wav_from_spectogram(file_path, generated_spectrogram, sr=44100)
   '''