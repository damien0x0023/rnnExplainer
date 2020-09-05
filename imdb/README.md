# Train a LSTM network

This directory includes code and data to train a LSTM network model
(inspired by the example in [tensorflow.tfjs-examples](https://github.com/tensorflow/tfjs-examples)
on sentiment analysit from the [Keras](https://keras.io/api/datasets/#imdb-movie-reviews-sentiment-classification).

## Training your own sentiment analysis model in tfjs-node

To train the model using tfjs-node, do

```sh
yarn
yarn train <MODEL_TYPE>
```

where `MODEL_TYPE` is a required argument that specifies what type of model is to be
trained. The available options are:

- `multihot`: A model that takes a multi-hot encoding of the words in the sequence.
  In terms of data representation and model complexity, this is the simplest model
  in this example.
- `flatten`: A model that flattens the embedding vectors of all words in the sequence.
- `cnn`: A 1D convolutional model, with a dropout layer included.
- `simpleRNN`: A model that uses a SimpleRNN layer (`tf.layers.simpleRNN`)
- `lstm`: A model that uses a LSTM laayer (`tf.layers.lstm`)
- `bidirectionalLSTM`: A model that uses a bidirectional LSTM layer
  (`tf.layers.bidirectional` and `tf.layers.lstm`)

By default, the training happens on the CPU using the Eigen kernels from tfjs-node.
You can make the training happen on GPU by adding the `--gpu` flag to the command, e.g.,

```sh
yarn train --gpu <MODEL_TYPE>
```

The training process will download the training data and metadata form the web
if they haven't been downloaded before. After the model training completes, the model
will be saved to the `public/resources` folder, alongside a `metadata.json` file.

Other arguments of the `yarn train` command include:

- `--maxLen` allows you to specify the sequence length.
- `--numWords` allows you to specify the vocabulary size.
- `--embeddingSize` allows you to adjust the dimensionality of the embedding vectors.
- `--epochs`, `--batchSize`, and `--validationSplit` are training-related settings.
- `--modelSavePath` allows you to specify where to store the model and metadata after
  training completes.
- `--embeddingFilesPrefix` Prefix for the path to which to save the embedding vectors
  and labels files (optinal). See the section below for details.
- `--logDir` This optional string lets you log the loss and accuracy values to
  a tensorboard log directory during training. For example if you start your training
  with command:

  ```sh
  yarn train lstm --logDir /tmp/my_lstm_logs
  ```

  You can use the following commands to start a tensorboard server in a separate
  terminal:

  ```sh
  pip install tensorboard   # Unless tensorboard is already installed
  tensorboard --logdir /tmp/my_lstm_logs
  ```

  Then you can open a browser tab and navigate to the http:// URL indicated by
  tensorboard (by default: http://localhost:6006) to view the loss and accuracy
  curves.

  You may encounter error info from node-pre-gyp that "This Node instance does not 
  support builds for N-API version 4" because the library deinitely build on N-API 4.
  This is a known issue of installing tfjs-node on windows with certain version of node.
  Node v10.16.3 should work or another command you can try is:
  ```sh
  npm rebuild @tensorflow/tfjs-node --build-from-source
  ```
