/* global tf */

import {OOV_INDEX, padSequences} from '../../imdb/sequence_utils';

// Network input image size
const networkInputSize = 64;

let indexFrom;
let maxLen;
let wordIndex;
let vocabularySize;

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

class Node {
  /**
   * Class structure for each neuron node.
   * 
   * @param {string} layerName Name of the node's layer.
   * @param {int} index Index of this node in its layer.
   * @param {string} type Node type {input, embedding, lstm, fc}. 
   * @param {number} bias The bias assocated to this node.
   * @param {number[]} output Output of this node.
   */
  constructor(layerName, index, type, bias, output) {
    this.layerName = layerName;
    this.index = index;
    this.type = type;
    this.bias = bias;
    this.output = output;

    // Weights are stored in the links
    this.inputLinks = [];
    this.outputLinks = [];
  }
}

class Link {
  /**
   * Class structure for each link between two nodes.
   * 
   * @param {Node} source Source node.
   * @param {Node} dest Target node.
   * @param {number} weight Weight associated to this link. It can be a number,
   *  1D array, or 2D array.
   */
  constructor(source, dest, weight) {
    this.source = source;
    this.dest = dest;
    this.weight = weight;
  }
}

/**
 * Load metadata file.
 *
 * @return An object containing metadata as key-value pairs.
 */
async function loadMetadata(url) {
  console.log('Loading metadata from ' + url)
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    console.log('Done loading metadata from '+url);

    indexFrom = metadata['index_from'];
    maxLen = metadata['max_len'];
    wordIndex = metadata['word_index'];
    vocabularySize = metadata['vocabulary_size'];
    console.log('indexFrom = ' , indexFrom);
    console.log('maxLen = ' , maxLen);
    // console.log('wordIndex = ' , wordIndex);
    console.log('vocabularySize = ', vocabularySize);

    return metadata;
  } catch(err) {
    console.error(err);
    console.log('Loading metadata failed.');
  }
}

/**
 * Get the 2D value array of the given review content.
 * 
 * @param {string} inputReview content of movie review
 * @returns A promise with the corresponding 2D array
 */
const getInputTextArray = (inputReview) => {
  // Convert to lower case and remove all punctuations.
  let inputText =
    inputReview.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split(' ');

    // Convert the words to a sequence of word indices.
  let sequence = inputText.map(word => {
      let this_wordIndex = wordIndex[word] + indexFrom;
      if (this_wordIndex > vocabularySize) {
        this_wordIndex = OOV_INDEX;
      }
      return this_wordIndex;
  });

  // Perform truncation and padding.
  let paddedSequence = padSequences([sequence], maxLen);
  console.log('paddedSequence is: ',paddedSequence);
  let tensor = tf.tensor2d(paddedSequence, [1,maxLen]);
  return tensor;
}

/**
 * return a object of elapsed time and final score
 * 
 * @param {Tensor} input Loaded input text tensor.
 * @param {Model} model Loaded tf.js model.
 */
const predictResult = (input, model) => {
  let beginMs = performance.now();
  let predictOut = model.predict(input);
  let score = predictOut.dataSync()[0];
  predictOut.dispose();
  let endMs = performance.now();

  return {score: score, elapsed: (endMs - beginMs)};
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
  let inputShape = model.layers[0].batchInputShape.slice(1);
  let inputTextArray = inputTextTensor.arraySync();

  // First layer's 100 nodes' outputs are the words of inputImageArray?
  for (let i = 0; i < inputShape[0]; i++) {
    let node = new Node('input', i, nodeType.INPUT, 0, inputTextArray[i]);
    inputLayer.push(node);
  }
                                                                                                                   
  rnn.push(inputLayer);
  let curLayerIndex = 1;

  for (let l = 0; l < model.layers.length; l++) {
    let layer = model.layers[l];
    // Get the current output
    let outputs = allOutputs[l].squeeze();
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
      curLayerType = nodeType.DENSE;
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
       
        // The new order is [output_dim, input_dim]
        let weights = layer.embeddings.val.transpose([1,0]).arraySync();

        for (let i=0; i<outputs.length; i++) {
          let node = new Node(layer.name, i, curLayerType, 
            bias, outputs[i]);
          
          // One-to-multiple links
          for (let j = 0; j < rnn[curLayerIndex -1].length; j++){
            let preNode = rnn[curLayerIndex -1][j];
            let curLink = new Link(preNode, node, weights[i][j]);
            preNode.outputLinks.push(curLink);
            node.inputLinks.push(curLink);
          }
          
          curLayerNodes.push(node);
        }

        break;
      }
      case nodeType.LSTM: {
        let bias = 0;
        let weight = null;

        break;
      }
      case nodeType.DENSE: {
        let bias = 0;
        let weight = null;

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

  return rnn;
}

/**
 * Construct a RNN with given metadata, model and input.
 * 
 * @param {string} inputMovieReview movie review.
 * @param {string} metadataFile filename and path of metadata.
 * @param {Model} model Loaded tf.js model.
 */
export const constructRNN = async (inputMovieReview, metadataFile, model) => {
  // load metadata of the pretrained model
  let sentimentMetadata = await loadMetadata(metadataFile);

  console.log('input review is: ', inputMovieReview)
  let inputTextTensor = await getInputTextArray(inputMovieReview);

  // let inputTextTensorBatch = tf.stack([inputTextTensor]);
  console.log('input text tensor is: ' + inputTextTensor);

  let preTensor = inputTextTensor; 
  let outputs = [];

  for (let l = 0; l< model.layers.length; l++) {
    let curTensor = model.layers[l].apply(preTensor);
    console.log('current layer name is: ', model.layers[l].name);

    let output = curTensor.squeeze();
    if (output.shape.length === 2 ) {
      console.log(output.shape);
      output = output.transpose([1, 0]);
      console.log(output.shape);
    }
    outputs.push(output);

    preTensor = curTensor;
  }
  console.log('final rnn outputs is ' )
  console.log(outputs);
  console.log('rnn result is ' + outputs[2])
  
  // let result = predictResult(inputTextTensor, model);
  // console.log('predicted time is: ', result.elapsed);
  // console.log('predicted score is: ', result.score);

  let rnn = constructRNNFromOutputs(outputs, model, inputTextTensor);
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
