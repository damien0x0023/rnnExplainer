export class Node {
    /**
     * Class structure for each neuron node.
     * 
     * @param {string} layerName Name of the node's layer.
     * @param {int} index Index of this node in its layer.
     * @param {string} type Node type {input, conv, pool, relu, fc}. 
     * @param {number} bias The bias assocated to this node.
     * @param {[[number]]} output Output of this node.
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
  
  export class Link {
    constructor(source, dest, weight) {
      this.source = source;
      this.dest = dest;
      this.weight = weight;
    }
  }