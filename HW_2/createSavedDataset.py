import pandas as pd
import numpy as np
import tensorflow as tf

# First load the pickle containing the dictionary with the 2 Pandas Dataframes
data = pd.read_pickle('appml-assignment1-dataset-v2.pkl')

# Create independent copies of the Dataframes
X, y = data['X'].copy(), data['y'].copy()

# Calculate the fractional change of the next hour's CAD-high and the 
# current hour's CAD-close
cad_close = X['CAD-close']
frac_change = (y-cad_close) / cad_close

# Define the bin edges from -0.001 to 0.001, with 21 evenly spaced boundaries


# Quantize the fractional change of the next hour’s CAD-high
# versus the previous hours CAD-close into 22 bins, based on 
# 21 evenly spaced boundaries stretching from −.001
# to .001 and including bins for being below −.001 and 
# above .001. 
bins = np.linspace(-0.001, 0.001, 21)
labels = np.digitize(frac_change, bins)

# Create temporal attributes for weekday, hour, and month
X['date'] = pd.to_datetime(X['date'])
weekday = X['date'].dt.weekday  # 0 = Monday
hour = X['date'].dt.hour        # 0–23
month = X['date'].dt.month      # 1–12

# Create a feature tickers.
# An array containing 188 numeric values of this instance excluding the date.
tickers_data = X.drop(columns=['date']).to_numpy(dtype=np.float32)
assert tickers_data.shape[1] == 188, "Expected 188 ticker features"

# Function to Create features for weekday, hour, month, and target labels
# Serialize each example object to into a string
def serialize_example(tickers, wd, hr, mo, target):
    """Create a serialized example for TFRecord.
    Args:
        tickers: A numpy array of ticker values.
        wd: Weekday as an integer.
        hr: Hour as an integer.
        mo: Month as an integer.
        target: Target label as an integer.
    Returns:
        A serialized string of the example."""
    feature = {
        'tickers': tf.train.Feature(float_list=tf.train.FloatList(value=tickers)),
        'weekday': tf.train.Feature(int64_list=tf.train.Int64List(value=[wd])),
        'hour': tf.train.Feature(int64_list=tf.train.Int64List(value=[hr])),
        'month': tf.train.Feature(int64_list=tf.train.Int64List(value=[mo])),
        'target': tf.train.Feature(int64_list=tf.train.Int64List(value=[target]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

with tf.io.TFRecordWriter('dataset.tfrecords') as writer:
    for i in range(len(X)):
        example = serialize_example(
            tickers=tickers_data[i],
            wd=int(weekday.iloc[i]),
            hr=int(hour.iloc[i]),
            mo=int(month.iloc[i]),
            target=int(labels[i])
        )
        writer.write(example)

print("TFRecord file 'dataset.tfrecords' created successfully.")