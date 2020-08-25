# RNN Explainer

An interactive visualization application designed to help non-experts learn about Recurrent Neural Networks (RNNs). This version is editted by CNN Explainer from Poloclub. The following is the sources:

[![Build Status](https://travis-ci.com/poloclub/cnn-explainer.svg?branch=master)](https://travis-ci.com/poloclub/cnn-explainer)
[![arxiv badge](https://img.shields.io/badge/arXiv-2004.15004-red)](http://arxiv.org/abs/2004.15004)

<!-- <a href="https://youtu.be/HnWIHWFbuUQ" target="_blank"><img src="https://i.imgur.com/TIKlgt6.png" style="max-width:100%;"></a> -->

For more information, check out their manuscript:

[**CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization**](https://arxiv.org/abs/2004.15004).
Wang, Zijie J., Robert Turko, Omar Shaikh, Haekyu Park, Nilaksh Das, Fred Hohman, Minsuk Kahng, and Duen Horng Chau.
arXiv preprint 2020. arXiv:2004.15004.


## Live Demo

For a live demo of their App, visit: http://poloclub.github.io/cnn-explainer/

## Running Locally

Clone or download this repository:

```
git clone git@github.com:damien0x0023/rnnExplainer.git

# use degit if you don't want to download commit histories
degit damien0x0023/rnnExplainer
```

Install the dependencies:

```
npm install 
```
or
```
yarn
```

Then run RNN Explainer:

```
npm run dev
```
or
```
yarn dev
```

Navigate to [localhost:5000](https://localhost:5000). You should see the Explainer running in your broswer.

To see how we trained the CNN or RNN, visit the directory [`./tiny-vgg/`](tiny-vgg) or [`./imdb/`](imdb).


## Training your own model in tfjs-node

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

## Credits


## License
The software is available under the [MIT License](https://github.com/poloclub/cnn-explainer/blob/master/LICENSE).


