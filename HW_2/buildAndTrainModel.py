import tensorflow as tf
from customImputerLayerDefinition import ImputerLayer
from sklearn.model_selection import train_test_split


# Define the parse example 
feature_description = {
    'tickers': tf.io.FixedLenFeature([188], tf.float32),
    'weekday': tf.io.FixedLenFeature([], tf.int64),
    'hour': tf.io.FixedLenFeature([], tf.int64),
    'month': tf.io.FixedLenFeature([], tf.int64),
    'target': tf.io.FixedLenFeature([], tf.int64)
}

# 1. Define parse_example function
def parse_example(example_proto):
    """ Parse a single example from the TFRecord file.
    Args:
        example_proto: A serialized example from the TFRecord file.
    Returns:
        A dictionary of features and the target label."""
        
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    target = parsed.pop('target')  # Remove 'target' for input features
    return parsed, target

# 2. Load and preprocess the dataset
batch_size = 32

# Load the dataset from the TFRecord file
raw_dataset = tf.data.TFRecordDataset("dataset.tfrecords")
parsed_dataset = raw_dataset.map(parse_example)

# Cache the result in memory using the cache command
parsed_dataset = parsed_dataset.cache().batch(batch_size)

# Convert dataset to a list of items to split
parsed_list = list(parsed_dataset.unbatch().as_numpy_iterator())

X_total, y_total = zip(*parsed_list)

# 3. First split the total dataset 70% train then split the remaining 30% into validation and test sets
# X_train, X_val, X_test are a list of dictionaries with keys 'tickers', 'weekday', 'hour', 'month'
# y_train, y_val, y_test are the corresponding target labels, a list of integers
X_train, X_temp, y_train, y_temp = train_test_split(X_total, y_total, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) 

def wrap_to_tf_dataset(X, y, shuffle=False):
    """ Wrap the features and labels into a TensorFlow dataset.
    For compatibility with Keras. Added shuffle option for regularization.
    Args:
        X: A list of dictionaries with features.
        y: A list of target labels.
        shuffle: Whether to shuffle the dataset.
    Returns:
        A TensorFlow dataset ready for training
        (dictionary of features, labels).
    """ 
    X_dict = {key: tf.convert_to_tensor([x[key] for x in X]) for key in X[0].keys()}
    dataset = tf.data.Dataset.from_tensor_slices((X_dict, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_data = wrap_to_tf_dataset(X_train, y_train, shuffle=True)
val_data = wrap_to_tf_dataset(X_val, y_val)
test_data = wrap_to_tf_dataset(X_test, y_test)

# 4. Create a series of 4 keras inputs for the 4 labels in the instance library
# Specifying the appropriate datatype and shape.

tickers_input = tf.keras.Input(shape=(188,), dtype=tf.float32,  name="tickers")
weekday_input = tf.keras.Input(shape=(), dtype=tf.int64, name="weekday")
hour_input = tf.keras.Input(shape=(), dtype=tf.int64, name="hour")
month_input = tf.keras.Input(shape=(), dtype=tf.int64, name="month")


# 5. Import the ImputerLayer  and adapt it to the training data for the ticker
# attributes.

# Create a single tensor for all the training tickers
tickers_train_tensor = tf.stack([tf.convert_to_tensor(x['tickers']) for x in X_train])
imputer = ImputerLayer()
imputer.adapt(tickers_train_tensor)

# 6. Create Normalization Layer and adapt it

imputed_tickers = imputer(tickers_train_tensor)
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(imputed_tickers)

# 7. Apply Imputer + Normalizer 

x_tickers = normalizer(imputer(tickers_input))

# 3.3. Build, Train, Evaluate, and Save the Keras Model

# 1. Feed the categorical attributes (weekday, hour, month) into embedding layers

weekday_embed = tf.keras.layers.Embedding(input_dim=7, output_dim=4)(weekday_input)
hour_embed = tf.keras.layers.Embedding(input_dim=24, output_dim=4)(hour_input)
month_embed = tf.keras.layers.Embedding(input_dim=13, output_dim=4)(month_input)


# 2. Concatenate the outputs of embedding layers and the normalizer to form 
# an input to following layers using tf.concat
# Flatten embeddings before concatenation to ensure correct shape
x_weekday = tf.keras.layers.Flatten()(weekday_embed)
x_hour = tf.keras.layers.Flatten()(hour_embed)
x_month = tf.keras.layers.Flatten()(month_embed)

x = tf.keras.layers.Concatenate(axis=-1)([x_tickers, x_weekday, x_hour, x_month])

# Create Early Stopping callback to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True, verbose=1)

# 3. Add subsequent layers to the model but with a final output layer 
# with 22 outputs corressponding to the probabilities of 22 bins of the fractional
# change.
# The Functional API structure is used to allow for more flexibility for the
# multiple differently-processed inputs.

x = tf.keras.layers.Dense(256, activation='relu')(x)
# Add dropout layers for regularization
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)  
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Final output layer with 22 outputs for the bins
output = tf.keras.layers.Dense(22, activation='softmax', name='output')(x)

# 4. Define the Keras Model
model = tf.keras.Model(
    inputs=[tickers_input, weekday_input, hour_input, month_input],
    outputs=output  
)

# 5. Compile and train the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # Use sparse if labels are integers
    metrics=['accuracy']
)
model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,  # Adjust epochs as needed
    batch_size=batch_size,
    callbacks=[early_stopping]
)

# 6. Save the model
model.export("mySavedModel") 

print("Model training complete andsaved successfully.")