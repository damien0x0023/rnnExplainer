/* global tf */

import {OOV_INDEX, padSequences} from '../../imdb/sequence_utils';
import {Node, Link} from './class';

// Enum of node types
const nodeType = {
  INPUT: 'input',
  CONV: 'conv',
  POOL: 'pool',
  RELU: 'relu',
  FC: 'fc',
  FLATTEN: 'flatten',
  EMBEDDING: 'embedding',
  LSTM: 'lstm',
  DENSE: 'dense',
};

export class SentimentPredictor{
  /**
   * 
   * @param {string} urls 
   */
  async init(urls) {
    this.urls= urls;
    this.model = await loadTrainedModel_rnn(urls.model);
    this.metadata = await this.loadMetadata(urls.metadata);

    return this;
  }

  /**
 * Load metadata file.
 *
 * @return An object containing metadata as key-value pairs.
 */
  async loadMetadata(url) {
    console.log('Loading metadata from ' + url)
    try {
      const metadataJson = await fetch(url);
      const metadata = await metadataJson.json();
      console.log('Done loading metadata from '+url);

      this.indexFrom = metadata['index_from'];
      this.maxLen = metadata['max_len'];
      this.wordIndex = metadata['word_index'];
      this.vocabularySize = metadata['vocabulary_size'];
      console.log('indexFrom = ' , this.indexFrom);
      console.log('maxLen = ' , this.maxLen);
      // console.log('wordIndex = ' , this.wordIndex);
      console.log('vocabularySize = ', this.vocabularySize);

      return metadata;
    } catch(err) {
      console.error(err);
      console.log('Loading metadata failed.');
    }
  }

  getInputTextTensor () {
    // Convert the words to a sequence of word indices.
    try {
      let sequence = this.inputArray.map(word => {
          let this_wordIndex = this.wordIndex[word] + this.indexFrom;
          // the issue: 'OOV' to NaN has been solved: 'OOV' and 
          // other words outside the dictionary to 2 now
          if (!this_wordIndex || this_wordIndex > this.vocabularySize) {
            this_wordIndex = OOV_INDEX;
          }
          return this_wordIndex;
      });
    
      // Perform truncation and padding.
      this.paddedSequence = padSequences([sequence], this.maxLen);
      // console.log('paddedSequence is: ', this.paddedSequence);
      let tensor = tf.tensor2d(this.paddedSequence, [1, this.maxLen]);
    
      return tensor
    } catch(err) {
      console.error(err);
      console.log('Get Input Text Tensor failed.');
    }
  }

  /**
 * return a object of elapsed time and final score
 * 
 * @param {Tensor} input Loaded input text tensor.
 * @param {Model} model Loaded tf.js model.
 */
  predictResult(inputMovieReview, model=this.model) {
    console.log("-----------------predict directly and print the result-----------------")
    let ipArray = getInputTextArray(inputMovieReview);
    if(!this.inputArray){
      this.inputArray = ipArray;
    } else if (this.inputArray !== ipArray){
      this.inputArray = ipArray;
    }

    let ipTensor = this.getInputTextTensor();
    if(!this.inputTensor){
      this.inputTensor = ipTensor;
    } else if (this.inputTensor !== ipTensor) {
      this.inputTensor = ipTensor;
    }

    // console.log('tensor is: '+ this.inputTensor);
    let beginMs = performance.now();
    let predictOut = model.predict(this.inputTensor);
    let res = predictOut.dataSync();
    let score = res[0];
    predictOut.dispose();
    let endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs), 
      inputReviewArray: this.inputArray,
      inputReviewTensor: this.inputTensor};
  }

  async constructNN(inputMovieReview, model= this.model) {
    console.log("-----------------predict layer by layer and generate NN structure-----------------")
    // console.log('input review is: ', inputMovieReview)

    // Get the array and tensor if do not execure predictOut before
    let ipArray = await getInputTextArray(inputMovieReview);
    if(!this.inputArray){
      this.inputArray = ipArray;
    } else if (this.inputArray !== ipArray){
      this.inputArray = ipArray;
    }
    // console.log('input text array is: ', this.inputArray);

    let ipTensor = await this.getInputTextTensor();
    if(!this.inputTensor){
      this.inputTensor = ipTensor;
    } else if (this.inputTensor !== ipTensor) {
      this.inputTensor = ipTensor;
    }

    // let inputTensorBatch = tf.stack([inputTensor]);
    console.log(this.inputTensor);

    let preTensor = this.inputTensor; 
    let outputs = [];

    for (let l = 0; l< model.layers.length; l++) {
      console.log('current layer name is: ', model.layers[l].name);
      let curTensor = model.layers[l].apply(preTensor);
      // console.log(curTensor);

      // Set the squeeze dim is 0 to unpack the batch otherwise it will 
      // ignore the final outcome if there is only one value.
      let output = curTensor.squeeze([0]);
      // let output = curTensor.squeeze();


      // if (output.shape.length === 2) {
      //   // console.log(output.shape);
      //   output = output.transpose([1, 0]);
      // } 
      console.log(output.shape);
      outputs.push(output);

      preTensor = curTensor;
    }
    console.log('final rnn outputs is ' )
    console.log(outputs);
    console.log('rnn result is ' + outputs[2])

    let rnn = constructRNNFromOutputs(outputs, model, this.inputTensor);
    return rnn;
  }
}

/**
 * Get the 1D value array of the given review content.
 * 
 * @param {string} inputReview content of movie review
 * @returns A promise with the corresponding 1D array
 */
const getInputTextArray = (inputReview) => {
  // Convert to lower case and remove all punctuations and more spaces.
  return inputReview.trim().toLowerCase()
      .replace(/(\.|\,|\!|\?|\\|\/|\-|\@|\#|\$|\%|\^|\&|\*|\(|\)|\+|\_|\=|\<|\>|\:|\;)/g, ' ')
      .replace(/\s+/g, ' ').trim().split(' ');
}

/**
 * Construct layer architecture of a RNN with given extracted outputs from every layer.
 * 
 * @param {number[][]} allOutputs Array of outputs for each layer.
 *  allOutputs[i][j] is the output for layer i node j.
 * @param {Model} model Loaded tf.js model.
 * @param {Tensor} inputTextTensor Loaded input text tensor.
 */
const constructRNNFromOutputs = (allOutputs, model, inputTextTensor) => {
  let rnn = [];

  // Add the first layer (input layer)
  let inputLayer = [];
  let nonPadInputLayer = [];
  let inputShape = model.layers[0].batchInputShape.slice(1);
  let inputTextArray = inputTextTensor.transpose([1,0]).arraySync();

  // First layer's 100 nodes' outputs are the words of inputImageArray?
  for (let i = 0; i < inputShape[0]; i++) {
    let node = new Node('input', i, nodeType.INPUT, 0, inputTextArray[i]);
    inputLayer.push(node);
    if (inputTextArray[i][0] !== 0) {
      nonPadInputLayer.push(node);
    }
  }
                                                                                                                   
  rnn.push(inputLayer);
  let curLayerIndex = 1;

  for (let l = 0; l < model.layers.length; l++) {
    let layer = model.layers[l];
    let outputs = null;
    // Get the current output, squeeze again if the tensor has more than two dims
    if (allOutputs[l].shape.length > 1) {
      outputs = allOutputs[l].squeeze();
    } else {
      outputs = allOutputs[l];
    }
    outputs = outputs.arraySync();

    let curLayerNodes = [];
    let curLayerType;
    
    // Identify layer type based on the layer name
    
    if (layer.name.includes('conv')) {
      curLayerType = nodeType.CONV;
    } else if (layer.name.includes('pool')) {
      curLayerType = nodeType.POOL;
    } else if (layer.name.includes('relu')) {
      curLayerType = nodeType.RELU;
    } else if (layer.name.includes('output')) {
      curLayerType = nodeType.FC;
    } else if (layer.name.includes('flatten')) {
      curLayerType = nodeType.FLATTEN;
    } else if (layer.name.includes('embedding')) {
      curLayerType = nodeType.EMBEDDING;
    } else if (layer.name.includes('lstm')) {
      curLayerType = nodeType.LSTM;
    } else if (layer.name.includes('dense')) {
      curLayerType = nodeType.DENSE; // or FC(fullyConnect)
    } else {
      console.log('Find unknown type');
    }

    // Construct this layer based on its layer type
    switch (curLayerType) {
      case nodeType.CONV: {
        let biases = layer.bias.val.arraySync();
        // The new order is [output_depth, input_depth, height, width]
        let weights = layer.kernel.val.transpose([3, 2, 0, 1]).arraySync();

        // Add nodes into this layer
        for (let i = 0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, biases[i],
            outputs[i]);

          // Connect this node to all previous nodes (create links)
          // CONV layers have weights in links. Links are one-to-multiple.
          for (let j = 0; j < rnn[curLayerIndex - 1].length; j++) {
            let preNode = rnn[curLayerIndex - 1][j];
            let curLink = new Link(preNode, node, weights[i][j]);
            preNode.outputLinks.push(curLink);
            node.inputLinks.push(curLink);
          }
          curLayerNodes.push(node);
        }
        break;
      }
      case nodeType.FC: {
        let biases = layer.bias.val.arraySync();
        // The new order is [output_depth, input_depth]
        let weights = layer.kernel.val.transpose([1, 0]).arraySync();

        // Add nodes into this layer
        for (let i = 0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, biases[i],
            outputs[i]);

          // Connect this node to all previous nodes (create links)
          // FC layers have weights in links. Links are one-to-multiple.

          // Since we are visualizing the logit values, we need to track
          // the raw value before softmax
          let curLogit = 0;
          for (let j = 0; j < rnn[curLayerIndex - 1].length; j++) {
            let preNode = rnn[curLayerIndex - 1][j];
            let curLink = new Link(preNode, node, weights[i][j]);
            preNode.outputLinks.push(curLink);
            node.inputLinks.push(curLink);
            curLogit += preNode.output * weights[i][j];
          }
          curLogit += biases[i];
          node.logit = curLogit;
          curLayerNodes.push(node);
        }

        // Sort flatten layer based on the node TF index
        rnn[curLayerIndex - 1].sort((a, b) => a.realIndex - b.realIndex);
        break;
      }
      case nodeType.RELU:
      case nodeType.POOL: {
        // RELU and POOL have no bias nor weight
        let bias = 0;
        let weight = null;

        // Add nodes into this layer
        for (let i = 0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, bias, outputs[i]);

          // RELU and POOL layers have no weights. Links are one-to-one
          let preNode = rnn[curLayerIndex - 1][i];
          let link = new Link(preNode, node, weight);
          preNode.outputLinks.push(link);
          node.inputLinks.push(link);

          curLayerNodes.push(node);
        }
        break;
      }
      case nodeType.FLATTEN: {
        // Flatten layer has no bias nor weights.
        let bias = 0;

        for (let i = 0; i < outputs.length; i++) {
          // Flatten layer has no weights. Links are multiple-to-one.
          // Use dummy weights to store the corresponding entry in the previsou
          // node as (row, column)
          // The flatten() in tf2.keras has order: channel -> row -> column
          let preNodeWidth = rnn[curLayerIndex - 1][0].output.length,
            preNodeNum = rnn[curLayerIndex - 1].length,
            preNodeIndex = i % preNodeNum,
            preNodeRow = Math.floor(Math.floor(i / preNodeNum) / preNodeWidth),
            preNodeCol = Math.floor(i / preNodeNum) % preNodeWidth,
            // Use channel, row, colume to compute the real index with order
            // row -> column -> channel
            curNodeRealIndex = preNodeIndex * (preNodeWidth * preNodeWidth) +
              preNodeRow * preNodeWidth + preNodeCol;
          
          let node = new Node(layer.name, i, curLayerType,
              bias, outputs[i]);
          
          // TF uses the (i) index for computation, but the real order should
          // be (curNodeRealIndex). We will sort the nodes using the real order
          // after we compute the logits in the output layer.
          node.realIndex = curNodeRealIndex;

          let link = new Link(rnn[curLayerIndex - 1][preNodeIndex],
              node, [preNodeRow, preNodeCol]);

          rnn[curLayerIndex - 1][preNodeIndex].outputLinks.push(link);
          node.inputLinks.push(link);

          curLayerNodes.push(node);
        }

        // Sort flatten layer based on the node TF index
        curLayerNodes.sort((a, b) => a.index - b.index);
        break;
      }
      case nodeType.EMBEDDING: {
        let bias = 0;
        // [10000,64]
        let weights = layer.embeddings.val.arraySync();
       
        // // The new order is [output_dim, input_dim]
        // let weights = layer.embeddings.val.transpose([1,0]).arraySync();

        for (let i=0; i<outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, 
            bias, outputs[i]);
          // Embedding layer have weights according to the word token (input Node[i].output)
          let preNode = rnn[curLayerIndex-1][i];
          // same as node.output
          let curLink = new Link(preNode,node, weights[preNode.output]);
          preNode.outputLinks.push(curLink);
          node.inputLinks.push(curLink);

          
          // // One-to-one links
          // for (let j = 0; j < rnn[curLayerIndex -1].length; j++){
          //   let preNode = rnn[curLayerIndex -1][j];
          //   // todo: double check j and the values of weights
          //   let curLink = new Link(preNode, node, weights[i][j]);
          //   preNode.outputLinks.push(curLink);
          //   node.inputLinks.push(curLink);
          // }
          
          curLayerNodes.push(node);
        }

        break;
      }
      case nodeType.LSTM: {
        /**
        * Long-Short Term Memory layer - Hochreiter 1997.
        * This is an `RNN` layer consisting of one `LSTMCell`. It operates on a 
        * sequence of inputs. The shape of the input (not including the first,
        * batch dimension) needs to be at least 2-D, with the first dimension being
        * time steps. For example:
        *
        * ```js
        * const lstm = tf.layers.lstm({units: 8, returnSequences: true});
        *
        * // Create an input with 10 time steps.
        * const input = tf.input({shape: [10, 20]});
        * const output = lstm.apply(input);
        *
        * console.log(JSON.stringify(output.shape));
        * // [null, 10, 8]: 1st dimension is unknown batch size; 2nd dimension is the
        * // same as the sequence length of `input`, due to `returnSequences`: `true`;
        * // 3rd dimension is the `LSTMCell`'s number of units. 
        * 
        *    tf.layers.lstm({units: 32})
        * In our project , the input size is [100, 64], so the time steps are 100 as
        * there are 100 words in a review and 64 dimensions of each word,
        * but the setting for units is 32 without returnSequences, so it only returns 
        * the output of last time step. [null,null,32] => 32. 
        *   kernel: (input_dim, unit * 4) => (64, 128)
        *   recurrent_kernel: (unit, unit * 4) => (32,128)
        *   bias: (unit * 4) (128)
        * In the cell of lstm, the hidden size (unit) is 32 and the input_dim is 64, 
        * the magic number 4 is due to  forget, input, candidate and output gates.
        * There will be 64 * 32 * 4 +32 * 32 * 4+ 32 * 4 hyperparameters. 
        * 
        * According code review of keras.layers.recurrent.py
        * https://github.com/tensorflow/tensorflow/blob/c7933ce9f3f25e592bef5f9a243ac72c5fd76cd1/tensorflow/python/keras/layers/recurrent.py#L1583
        * the first [0: 1* self.units(32)] elements in recurrent_kernel are weights for input gate,
        * the second [32:64] are weights for forget gate,
        * [64:96] for candidate gate and [96:128] for output gate.
        * The biases and weights in kernel follow the same order.
        * The recurrent_kernel provides weights to calculate and deliver the long term 
        * memory (c) and short term memory(h or o). The kernel keep the weights for the 
        * calculation with new input.
        */

        // 
        let bias = 0;
        // [128]
        let kernelBiases = layer.cell.bias.val.arraySync();
        // let splitedBiases = tf.split(biases,4)
        // New order is [units*4, input_dim]
        // [64, 128] => [128,64]
        let kernelWeights = layer.cell.kernel.val.transpose([1, 0]).arraySync();
        // [32, 128] => [128.32]
        let recurrent_kernel_weights = layer.cell.recurrentKernel.val.transpose([1,0]).arraySync();
        // There is only one node here and the output is just for the last time step, 
        // so we cannot use outputs for nodes at different time step directly.
        let node = new Node(layer.name, 0, curLayerType, bias, outputs);

        for (let j=0; j<rnn[curLayerIndex-1].length; j++) {
          let preNode =rnn[curLayerIndex-1][j];
          // according to code review: 
          // h_tm1 = states[0]  # previous memory state
          // c_tm1 = states[1]  # previous carry state
          // ...
          //   k_i, k_f, k_c, k_o = array_ops.split(
          //     self.kernel, num_or_size_splits=4, axis=1)
          //   x_i = K.dot(inputs_i, k_i)
          //   x_f = K.dot(inputs_f, k_f)
          //   x_c = K.dot(inputs_c, k_c)
          //   x_o = K.dot(inputs_o, k_o)
          // the kernelweights are for the inputs from embbeding layer
          // ...
          //   i = self.recurrent_activation(
          //     x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
          //   f = self.recurrent_activation(x_f + K.dot(
          //     h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
          //   c = f * c_tm1 + i * self.activation(x_c + K.dot(
          //     h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
          //   o = self.recurrent_activation(
          //     x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
          //   let curLink = new Link(preNode, node , kernelWeights)
          // the recurrentKernelWeights are for the previous memory state
          let curLink = new Link(preNode, node, kernelWeights);
          node.inputLinks.push(curLink);
        }
        curLayerNodes.push(node);
        
        // //add nodes into this layer
        // for (let i=0; i < outputs.length; i++){
        //   let node = new Node(layer.name, i, curLayerType, 
        //     biases[i], outputs[i]);

        //   // Connect this node to all previous nodes (create links)
        //   // LSTM layers have weights in links. Links are one-to-multiple.
        //   for (let j=0; j < rnn[curLayerIndex -1].length; j++) {
        //       let preNode = rnn[curLayerIndex-1][j];
        //       let curLink = new Link(preNode, node, weights[i][j]);
        //       preNode.outputLinks.push(curLink);
        //       node.inputLinks.push(curLink);
        //     }
        //   curLayerNodes.push(node);
        // }
        break;
      }
      case nodeType.DENSE: {
        let biases = layer.bias.val.arraySync();
        let weights = layer.kernel.val.transpose([1,0]).arraySync();

        // add nodes into this layer
        for (let i =0; i < outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType,
            biases[i], outputs[i]);
          
          // only the output of LSTM at the last time step exists
          // Since we are visualizing the logit values, we need to track
          // the raw value before ...
          let curLogit = 0;
          for (let j = 0; j < rnn[curLayerIndex - 1].length; j++) {
            let preNode = rnn[curLayerIndex - 1][j];
            let curLink = new Link(preNode, node, weights[i]);
            preNode.outputLinks.push(curLink);
            node.inputLinks.push(curLink);
            for (let k = 0; k<preNode.output.length; k++)
            curLogit += preNode.output[k] * weights[j][k];              
            }
          curLogit += biases[i];
          node.logit = curLogit;
          curLayerNodes.push(node);
        }

        // Sort flatten layer based on the node TF index
        rnn[curLayerIndex - 1].sort((a, b) => a.realIndex - b.realIndex);

        break;
      }
      default:
        console.error('Encounter unknown layer type');
        break;
    }

    // Add current layer to the NN
    rnn.push(curLayerNodes);
    curLayerIndex++;
  }

  rnn.nonPadInput = nonPadInputLayer;
  return rnn;
}

/**
 * Wrapper to load a model.
 * 
 * @param {string} modelFile Filename of converted (through tensorflowjs.py)
 *  model json file.
 */
export const loadTrainedModel_rnn = (modelFile) => {
  return tf.loadLayersModel(modelFile);
}
