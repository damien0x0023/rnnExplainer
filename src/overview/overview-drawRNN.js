/* global d3, SmoothScroll */

import { 
  svgStore_rnn, vSpaceAroundGapStore_rnn, hSpaceAroundGapStore_rnn, rnnStore,
  nodeCoordinateStore_rnn, selectedScaleLevelStore_rnn, rnnLayerRangesStore,
  detailedModeStore_rnn, rnnLayerMinMaxStore, hoverInfoStore_rnn, reviewArrayStore
} from '../stores.js';
import {
  getExtent, getLinkDataRNN
} from './drawRNN-utils.js';
import { rnnOverviewConfig } from '../config.js';

// Configs
const layerColorScales = rnnOverviewConfig.layerColorScales;
const nodeLength = rnnOverviewConfig.nodeLength;
const nodeHeight = rnnOverviewConfig.nodeHeight;
const embeddingLen = rnnOverviewConfig.embedddingLength;
const inputNodeHeight = rnnOverviewConfig.inputNodeHeight;
const numLayers = rnnOverviewConfig.numLayers;
const edgeOpacity = rnnOverviewConfig.edgeOpacity;
const edgeInitColor = rnnOverviewConfig.edgeInitColor;
const edgeStrokeWidth = rnnOverviewConfig.edgeStrokeWidth;
const svgPaddings = rnnOverviewConfig.svgPaddings;
const gapRatio = rnnOverviewConfig.gapRatio;
const classLists = rnnOverviewConfig.classLists;
const formater = d3.format('.4f');



// Shared variables
//for rnn
let svg_rnn = undefined;
svgStore_rnn.subscribe( value => {svg_rnn = value;} )

let vSpaceAroundGap_rnn = undefined;
vSpaceAroundGapStore_rnn.subscribe( value => {vSpaceAroundGap_rnn = value;} )

let hSpaceAroundGap_rnn = undefined;
hSpaceAroundGapStore_rnn.subscribe( value => {hSpaceAroundGap_rnn = value;} )

let rnn = undefined;
rnnStore.subscribe( value => {rnn = value;} )

let nodeCoordinate_rnn = undefined;
nodeCoordinateStore_rnn.subscribe( value => {nodeCoordinate_rnn = value;} )

let selectedScaleLevel_rnn = undefined;
selectedScaleLevelStore_rnn.subscribe( value => {selectedScaleLevel_rnn = value;} )

let rnnLayerRanges = undefined;
rnnLayerRangesStore.subscribe( value => {rnnLayerRanges = value;} )

let rnnLayerMinMax = undefined;
rnnLayerMinMaxStore.subscribe( value => {rnnLayerMinMax = value;} )

let detailedMode_rnn = undefined;
detailedModeStore_rnn.subscribe( value => {detailedMode_rnn = value;} )

let reviewArray = undefined;
reviewArrayStore.subscribe(value => {reviewArray = value;})

// There are 2 short gaps(1 left 1 right) and 3 long gaps (btw nodes)
let n_shortGaps = 3;
let n_longGaps = 2;
// for module view
let num_module = 2;
let num_stack = 1;

const enlargeOutputImage = (oriWidth, oriHeight, enlargedWidth, 
  enlargedHeight, bufferCan, bufferCtx,img) =>{
  // canvas.toDataURL() only exports image in 96 DPI, so we can hack it to have
  // higher DPI by rescaling the image using canvas magic
  let larCanvas = document.createElement('canvas');
  larCanvas.width = enlargedWidth ;
  larCanvas.height = enlargedHeight ;
  let larCanvasContext = larCanvas.getContext('2d');

  // Use drawImage to resize the original pixel array, and put the new image
  // (canvas) into corresponding canvas
  bufferCtx.putImageData(img, 0, 0);
  // drawImage(img,sx,sy,swidth,sheight,x,y,width,height);
  // sx Optional
  // The x-axis coordinate of the top left corner of the sub-rectangle of the source image to draw into the destination context.
  // sy Optional
  // The y-axis coordinate of the top left corner of the sub-rectangle of the source image to draw into the destination context.
  // sWidth Optional
  // The width of the sub-rectangle of the source image to draw into the destination context. If not specified, the entire rectangle from the coordinates specified by sx and sy to the bottom-right corner of the image is used.
  // sHeight Optional
  // The height of the sub-rectangle of the source image to draw into the destination context.
  // dx
  // The x-axis coordinate in the destination canvas at which to place the top-left corner of the source image.
  // dy
  // The y-axis coordinate in the destination canvas at which to place the top-left corner of the source image.
  // dWidth Optional
  // The width to draw the image in the destination canvas. This allows scaling of the drawn image. If not specified, the image is not scaled in width when drawn.
  // dHeight Optional
  // The height to draw the image in the destination canvas. This allows scaling of the drawn image. If not specified, the image is not scaled in height when drawn.
  
  // todo: the dimension of embbeding layer needs to be reviewed
  larCanvasContext.drawImage(bufferCan, 0, 0, oriWidth, oriHeight,
      0, 0, enlargedWidth, enlargedHeight);

  let imgURL = larCanvas.toDataURL();
  larCanvas.remove();

  return imgURL;
}

/**
 * Use bounded d3 data to draw one canvas
 * @param {object} d d3 data
 * @param {index} i d3 data index
 * @param {[object]} g d3 group
 * @param {number} range color range map (max - min)
 */
export const drawOutputRNN = (d, i, g, range) => {
  let image = g[i];
  //draw and update the colors of image nodes
  let colorScale = layerColorScales[d.type];

  // Set up a second canvas in order to resize image
  // imageLength is the width of canvas in rnn
  let imageLength = d.output.length === undefined ? 1 : d.output.length;

  let bufferCanvas = document.createElement("canvas");
  let bufferContext = bufferCanvas.getContext("2d");

  bufferCanvas.width = imageLength;
  bufferCanvas.height = imageLength;
  // the above code may cause problem because the imageHeight is not the same as imageLength 
  // the height of node for embedding unit in rnn is 1
  let imageHeight = 1 ;
  // if (!d.output.length || !d.output[0].length){
  //   // number or 1d array
  //   imageHeight = 1;
  // } else if (!d.output[0][0].length )  {
  //   // 2d array
  //   imageHeight = d.output[0].length;
  // } 
  // bufferCanvas.height = imageHeight;

  // Fill image pixel array
  let imageSingle = bufferContext.getImageData(0, 0, imageLength, imageHeight);
  let imageSingleArray = imageSingle.data;
  let imageDataURL;

  for (let j = 0; j< imageLength; j++){
    let color;
    if (d.type === 'input'){
      color = d3.rgb(colorScale(0.1 +d.output/range));
    } else if (d.type.includes('lstm')) {
      color = d3.rgb(colorScale((d.output+range/2)/range));
    } else if (d.type.includes('embedding')){
      color = d3.rgb(colorScale((d.output[j] + range/2)/range));
    }

    imageSingleArray[4*j] = color.r;
    imageSingleArray[4*j + 1] = color.g;
    imageSingleArray[4*j + 2] = color.b;
    imageSingleArray[4*j + 3] = 255;
  }
  
  if (d.type === 'input') {
    imageDataURL = enlargeOutputImage(imageLength, imageHeight, nodeLength,
      inputNodeHeight, bufferCanvas, bufferContext, imageSingle);
  } else if(d.type == 'embedding'){
    imageDataURL = enlargeOutputImage(imageLength, imageHeight, 3*embeddingLen,
      3*inputNodeHeight, bufferCanvas, bufferContext, imageSingle);
  } else{
    imageDataURL = enlargeOutputImage(imageLength, imageHeight, nodeLength,
      nodeHeight, bufferCanvas, bufferContext, imageSingle);
  }

  d3.select(image).attr('xlink:href', imageDataURL);

  // Destory the buffer canvas
  bufferCanvas.remove();
  // largeCanvas.remove();
}

/**
 * Draw bar chart to encode the output value
 * @param {object} d d3 data
 * @param {index} i d3 data index
 * @param {[object]} g d3 group
 * @param {function} scale map value to length
 */
const drawOutputScore = (d, i, g, scale) => {
  let group = d3.select(g[i]);
  // draw and update the width of output rects
  group.select('rect.output-rect')
    .transition('dense')
    .delay(500)
    .duration(800)
    .ease(d3.easeCubicIn)
    .attr('width', scale(d.output));
  // // draw and update the proba of outputs
  // group.select('text.annotation-output')
  //   .transition('dense')
  //   .delay(500)
  //   .duration(800)
  //   .ease(d3.easeCubicIn)
  //   .style('dominant-baseline', 'middle')
  //   .style('font-size', '11px')
  //   .style('fill', 'black')
  //   .style('opacity', 0.5)
  //   .text((d, i) => d.output.toFixed(4));
}

/**
 * Create color gradient for the legend
 * @param {[object]} g d3 group
 * @param {function} colorScale Colormap
 * @param {string} gradientName Label for gradient def
 * @param {number} min Min of legend value
 * @param {number} max Max of legend value
 */
const getLegendGradient = (g, colorScale, gradientName, min, max) => {
  if (min === undefined) { min = 0; }
  if (max === undefined) { max = 1; }
  let gradient = g.append('defs')
    .append('svg:linearGradient')
    .attr('id', `${gradientName}`)
    .attr('x1', '0%')
    .attr('y1', '100%')
    .attr('x2', '100%')
    .attr('y2', '100%')
    .attr('spreadMethod', 'pad');
  let interpolation = 10
  for (let i = 0; i < interpolation; i++) {
    let curProgress = i / (interpolation - 1);
    let curColor = colorScale(curProgress * (max - min) + min);
    gradient.append('stop')
      .attr('offset', `${curProgress * 100}%`)
      .attr('stop-color', curColor)
      .attr('stop-opacity', 1);
  }
}

const legendUpdate = (range, domain, levelName, layerIndex=-1,legendIndex=-1) =>{
    let curLegendScale = d3.scaleLinear()
      .range([0, range])
      .domain([-domain / 2, domain / 2]);
  
    let curLegendAxis = d3.axisBottom()
      .scale(curLegendScale)
      .tickFormat(d3.format('.2f'))
      .tickValues([-domain / 2, 0, domain / 2]);

    let selector;
    if (levelName === 'local') {
      selector = `g#local-legend-${layerIndex}-${legendIndex}`;
    } else if( levelName === 'module'){
      selector = `g#module-legend-${layerIndex}`;
    } else if (levelName === 'global') {
      selector = `g#global-legend`;
    }    

    svg_rnn.select(selector).select('g').call(curLegendAxis);
  }



const legendDrawer = (range, domain, levelName, legends, startIndex, 
    legendHeight, gradientParm = '', layerIndex=-1, legendIndex=-1) => {
  let curLegendScale = d3.scaleLinear()
    .range([0, range])
    .domain([-domain / 2, domain / 2]);

  let curLegendAxis = d3.axisBottom()
    .scale(curLegendScale)
    .tickFormat(d3.format('.2f'))
    .tickValues([-domain / 2, 0, domain / 2]);

  let id;
  if (levelName === 'local') {
    id = `local-legend-${layerIndex}-${legendIndex}`;
  } else if( levelName === 'module'){
    id = `module-legend-${layerIndex}`;
  } else if (levelName === 'global') {
    id = `global-legend`;
  }

  let curLegend = legends.append('g')
    .attr('class', `legend ${levelName}-legend`)
    .attr('id', id)
    .classed('hidden', !detailedMode_rnn || selectedScaleLevel_rnn !== levelName)
    .attr('transform', `translate(${nodeCoordinate_rnn[startIndex][0].x}, ${0})`);

  curLegend.append('g')
    .attr('transform', `translate(0, ${legendHeight - 3})`)
    .call(curLegendAxis)

  curLegend.append('rect')
    .attr('width', range)
    .attr('height', legendHeight)
    .style('fill', gradientParm);
}

/**
 * Draw all legends
 * @param {object} legends Parent group
 * @param {number} legendHeight Height of the legend element
 */
const drawAllLegends = (legends, legendHeight) => {
  // Add local legends
  for (let i = 0; i < num_stack; i++){
    let start = 1 + i * num_module;

    let range1 = 1 * embeddingLen + 0 * hSpaceAroundGap_rnn- 1.2;
    let domain1 = rnnLayerRanges.local[start];
    legendDrawer(range1, domain1, 'local', legends, start, legendHeight, 'url(#embeddingGradient)',i , 1);

    let range2 = 1 * nodeLength + 0 * hSpaceAroundGap_rnn- 1.2;
    let domain2= rnnLayerRanges.local[start + 1];
    legendDrawer(range2, domain2, 'local', legends, start+1, legendHeight, 'url(#lstmGradient)',i, 2);
  }

  // Add module legends
  for (let i = 0; i < num_stack; i++){
    let start = 1 + i * num_module;
    let domain = rnnLayerRanges.module[start];
    let range = (num_module-1) * nodeLength + embeddingLen+ 0 * hSpaceAroundGap_rnn +
              (num_module-1) * hSpaceAroundGap_rnn * gapRatio - 1.2;

    legendDrawer(range, domain,'module', legends, start, legendHeight, 'url(#lstmGradient)',i);
  }

  // Add global legends
  let start = 1;
  // let range = rnnLayerRanges.global[start];

  let domain = rnnLayerRanges.global[start];
  let range = (numLayers-3) * nodeLength
            + embeddingLen + (n_shortGaps-3) * hSpaceAroundGap_rnn 
            + (n_longGaps-1) * hSpaceAroundGap_rnn * gapRatio - 1.2
  
  legendDrawer(range, domain,'global', legends, start, legendHeight, 'url(#lstmGradient)');


  // Add output legend
  let outputRectScale = d3.scaleLinear()
        .domain([rnnLayerMinMax[nodeCoordinate_rnn.length-1].min, 
          rnnLayerMinMax[nodeCoordinate_rnn.length-1].max])
        .range([0, nodeLength - 1.2]);

  let outputLegendAxis = d3.axisBottom()
    .scale(outputRectScale)
    .tickFormat(d3.format('.1f'))
    .tickValues([0, rnnLayerMinMax[nodeCoordinate_rnn.length-1].max])
  
  let outputLegend = legends.append('g')
    .attr('class', 'legend output-legend')
    .attr('id', 'output-legend')
    .classed('hidden', !detailedMode_rnn)
    .attr('transform', `translate(${nodeCoordinate_rnn[nodeCoordinate_rnn.length-1][0].x}, ${0})`);
  
  outputLegend.append('g')
    .attr('transform', `translate(0, ${legendHeight - 3})`)
    .call(outputLegendAxis);

  outputLegend.append('rect')
    .attr('width', nodeLength - 0.3)
    .attr('height', legendHeight)
    .style('fill','gray');
  
  // Add input image legend
  let inputScale = d3.scaleLinear()
    .range([0, nodeLength - 1.2])
    .domain([0, rnnLayerRanges.local[0]]);

  let inputLegendAxis = d3.axisBottom()
    .scale(inputScale)
    // .tickFormat(d3.format('.1f'))
    .tickValues([0, rnnLayerRanges.local[0]]);

  let inputLegend = legends.append('g')
    .attr('class', 'legend input-legend')
    .classed('hidden', !detailedMode_rnn)
    .attr('transform', `translate(${nodeCoordinate_rnn[0][0].x}, ${0})`);
  
  inputLegend.append('g')
    .attr('transform', `translate(0, ${legendHeight - 3})`)
    .call(inputLegendAxis);

  inputLegend.append('rect')
    .attr('x', 0.3)
    .attr('width', nodeLength - 0.3)
    .attr('height', legendHeight)
    // .attr('transform', `rotate(180, ${nodeLength/2}, ${legendHeight/2})`)
    .style('stroke', 'rgb(20, 20, 20)')
    .style('stroke-width', 0.3)
    .style('fill', 'url(#inputGradient)');
}

/**
 * return the horizontal gap for the whole NN
 * 
 * @param {number} width 
 */
const calcHorSpaceGap = (width) => {
  return (width - nodeLength * (numLayers-1) - embeddingLen)  / (n_shortGaps + n_longGaps * gapRatio);
}

/**
 * return the vertical gap value for curLayer
 * 
 * @param {number} height 
 * @param {number} curLayerLength 
 * @param {number} curLayerNodeHeight 
 */
const calcVerSpaceGap = (height, curLayerLength, curLayerNodeHeight) => {
  return (height - curLayerNodeHeight * curLayerLength) / (curLayerLength + 1);
}

/** 
 *  add labels for each layer
 */
const addLabels = ()=> {
  let layerNames = rnn.map(d => {
    if (d[0].layerName === 'dense_Dense1' || d[0].layerName.includes('lstm')) {
      return {
        name: d[0].layerName,
        dimension: `(${d.length})`
      }
    } else {
      return {
        name: d[0].layerName,
        dimension: `(${d[0].output.length}, ${d.length})`
      }
    }
  });

  let svgHeight = Number(d3.select('#rnn-svg').style('height').replace('px', '')) + 150;
  let scroll = new SmoothScroll('a[href*="#"]', {offset: -svgHeight});
  
  let detailedLabels = svg_rnn.selectAll('g.layer-detailed-label')
    .data(layerNames)
    .enter()
    .append('g')
    .attr('class', 'layer-detailed-label')
    .attr('id', (d, i) => `layer-detailed-label-${i}`)
    .classed('hidden', !detailedMode_rnn)
    .attr('transform', (d, i) => {
      let x = !d.name.includes('embedding') 
        ? nodeCoordinate_rnn[i][0].x + nodeLength / 2
        : nodeCoordinate_rnn[i][0].x + embeddingLen / 2;
      let y = svgPaddings.top / 2 - 4 ;
      return `translate(${x}, ${y})`;
    })
    .style('cursor', d => d.name.includes('dense') ? 'default' : 'help')
    .on('click', (d) => {
      let target = '';
      if (d.name.includes('conv')) { target = 'convolution' }
      if (d.name.includes('relu')) { target = 'relu' }
      if (d.name.includes('max_pool')) { target = 'pooling'}
      if (d.name.includes('input')) { target = 'input'}
      if (d.name.includes('embedding')) { target = 'embedding'}
      if (d.name.includes('lstm')) { target = 'lstm'}  

      // Scroll to a article element
      let anchor = document.querySelector(`#article-${target}`);
      scroll.animateScroll(anchor);
    });
  
  detailedLabels.append('title')
    .text('Move to article section');
    
  detailedLabels.append('text')
    .style('opacity', 0.7)
    .style('dominant-baseline', 'middle')
    .append('tspan')
    .style('font-size', '12px')
    .text(d => d.name)
    .append('tspan')
    .style('font-size', '8px')
    .style('font-weight', 'normal')
    .attr('x', 0)
    .attr('dy', '1.5em')
    .text(d => d.dimension);
  
  let labels = svg_rnn.selectAll('g.layer-label')
    .data(layerNames)
    .enter()
    .append('g')
    .attr('class', 'layer-label')
    .attr('id', (d, i) => `layer-label-${i}`)
    .classed('hidden', detailedMode_rnn)
    .attr('transform', (d, i) => {
      let x = !d.name.includes('embedding') 
        ? nodeCoordinate_rnn[i][0].x + nodeLength / 2
        : nodeCoordinate_rnn[i][0].x + embeddingLen / 2;
      let y = svgPaddings.top / 2 - 6 ;
      return `translate(${x}, ${y})`;
    })
    .style('cursor', d => d.name.includes('dense') ? 'default' : 'help')
    .on('click', (d) => {
      let target = '';
      if (d.name.includes('conv')) { target = 'convolution' }
      if (d.name.includes('relu')) { target = 'relu' }
      if (d.name.includes('max_pool')) { target = 'pooling'}
      if (d.name.includes('input')) { target = 'input'}
      if (d.name.includes('embedding')) { target = 'embedding'}
      if (d.name.includes('lstm')) { target = 'lstm'}   

      // Scroll to a article element
      let anchor = document.querySelector(`#article-${target}`);
      scroll.animateScroll(anchor);
    });
  
  labels.append('title')
    .text('Move to article section');
  
  labels.append('text')
    .style('dominant-baseline', 'middle')
    .style('opacity', 0.8)
    .text(d => {
      if (d.name.includes('conv')) { return 'conv' }
      if (d.name.includes('relu')) { return 'relu' }
      if (d.name.includes('max_pool')) { return 'max_pool'}
      if (d.name.includes('embedding')) { return 'embedding'}
      if (d.name.includes('lstm')) { return 'lstm'}     
      if (d.name.includes('dense')) { return 'output'}   

      return d.name
    });
}

/**
 * add legends for layers with image nodes
 */
const addLegends = () => {
  getLegendGradient(svg_rnn, layerColorScales.embedding, 'embeddingGradient');
  getLegendGradient(svg_rnn, layerColorScales.lstm, 'lstmGradient');
  getLegendGradient(svg_rnn, layerColorScales.input, 'inputGradient');
  // getLegendGradient(svg_rnn, layerColorScales.dense, 'denseGradient');

  // same as input nodes
  let legendHeight = inputNodeHeight;
  // vSpaceAroundGap_rnn for each layer varies because of various number of nodes
  let legentY = svgPaddings.top + vSpaceAroundGap_rnn 
       * (rnn[rnn.length-1].length+1) + 3 * legendHeight;

  // y is based on the last one of vSpaceAroundGap_rnn
  let legends = svg_rnn.append('g')
      .attr('class', 'color-legend')
      .attr('transform', `translate(${0}, ${legentY})`);
  
  drawAllLegends(legends, legendHeight);
}

/**
 *  add edges btw layers
 * 
 * @param {object} rnnGroup Group to appen rnn elements to
 */
const addEdges = (rnnGroup)=> {
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  let linkData = getLinkDataRNN(nodeCoordinate_rnn, rnn);
  // console.log('linkData is: ', linkData);
  let edgeGroup = rnnGroup.append('g')
    .attr('class', 'edge-group');

  edgeGroup.selectAll('path.edge')
    .data(linkData)
    .enter()
    .append('path')
    .attr('class', d =>
      `edge edge-${d.targetLayerIndex} edge-${d.targetLayerIndex}-${d.targetNodeIndex}`)
    .attr('id', d => 
      `edge-${d.targetLayerIndex}-${d.targetNodeIndex}-${d.sourceNodeIndex}`)
    .attr('d', d => linkGen({source: d.source, target: d.target}))
    .style('fill', 'none')
    .style('stroke-width', d =>d.targetLayerIndex ===2 ? edgeStrokeWidth:edgeStrokeWidth*4)
    .style('opacity', edgeOpacity)
    .style('stroke', edgeInitColor);
}

/**
 * create image nodes, rects, texts and annotations for input layer
 * 
 * @param {*} nodeGroups 
 * @param {*} left 
 * @param {*} l 
 * @param {*} curLayer 
 */
const initInputLayer = (nodeGroups, left, l, curLayer) => {
  nodeGroups.append('image')
  .attr('class', 'node-image')
  .attr('width', nodeLength)
  .attr('height', inputNodeHeight)
  .attr('x', left)
  .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y)

  nodeGroups.append('rect')
    .attr('class', 'input-rect')
    .attr('class','bounding')
    .attr('width', nodeLength)
    .attr('height', inputNodeHeight)
    .attr('x', left)
    .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y)
    .style('fill', 'none')
    .style('stroke', 'gray')
    .style('stroke-width', 1)
    .classed('hidden', true);  

  // nodeGroups.append('text')
  //   .attr('class', 'input-text')
  //   .attr('x', 0)
  //   // .attr('x', svgPaddings.left/3)
  //   .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y)
  //   .style('dominant-baseline', 'middle')
  //   .style('font-size', '6px')
  //   .style('fill', 'black')
  //   .style('opacity', 0.8)
  //   .text((d, i) => d.output[0] === 0 
  //     ? `<pad>`:`${reviewArray[i + reviewArray.length - curLayer.length]}`);

  nodeGroups.append('text')
    .attr('class', 'input-text-index')
    .attr('x', left - 20)
    .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y+inputNodeHeight/2)
    .style('dominant-baseline', 'middle')
    .style('font-size', '6px')
    .style('fill', 'black')
    .style('opacity', 0.8)
    .text((d, i) => `#${i+1}`);

  nodeGroups.append('text')
    .attr('class','input-annotation')
    .attr('x', left + nodeLength/2)
    .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y+inputNodeHeight/2)
    .style('dominant-baseline', 'middle')
    .style('font-size', '6px')
    .style('fill', 'rgb(255,165,42)')
    .style('opacity', 0.8)
    // .text((d,i) => d.output);
    .text((d,i)=>d.output[0]);
}

/**
 * create rects, text and annotation for output layer
 * 
 * @param {*} nodeGroups 
 * @param {*} left 
 * @param {*} l 
 */
const initOutputLayer = (nodeGroups, left, l) => {
        // Add a rectangle to show the border
        nodeGroups.append('rect')
        .attr('class', 'bounding')
        .attr('x', left)
        .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y )
        .attr('height', nodeHeight)
        .attr('width', nodeLength)
        .style('fill', 'none')
        .style('stroke', 'gray')
        .style('stroke-width', 0.5)
        .classed('hidden', false);
        nodeGroups.append('rect')
          .attr('class', 'output-rect')
          .attr('x', left)
          .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y )
          .attr('height', nodeHeight)
          .attr('width', nodeLength)
          .style('fill', 'gray');
        // Add annotation text to tell readers the exact output probability
        // nodeGroups.append('text')
        //   .attr('class', 'annotation-output')
        //   .attr('id', (d, i) => `output-prob-${i}`)
        //   .attr('x', left + nodeLength/4)
        //   .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y + nodeLength / 2)
        //   // .style('dominant-baseline', 'middle')
        //   // .style('font-size', '11px')
        //   // .style('fill', 'black')
        //   // .style('opacity', 0.5)
        //   // .text((d, i) => d.output.toFixed(4));
        nodeGroups.append('text')
          .attr('class', 'output-text-one')
          .attr('x', left-nodeLength/5)
          .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y - nodeLength / 4)
          .style('dominant-baseline', 'middle')
          .style('font-size', '11px')
          .style('fill', 'black')
          .style('opacity', 0.5)
          .text((d, i) => classLists[i]);
        nodeGroups.append('text')
          .attr('class', 'output-text-two')
          .attr('x', left+nodeLength*4/5)
          .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y - nodeLength / 4)
          .style('dominant-baseline', 'middle')
          .style('font-size', '11px')
          .style('fill', 'black')
          .style('opacity', 0.5)
          .text((d, i) => classLists[i+1]);
}

/**
 * draw rects for output and display the scores
 */
const drawOutPutLayer = () => {
  // Compute the scale of the output score width (mapping the the node
  // width to the max output score)
  let outputRectScale = d3.scaleLinear()
        .domain([rnnLayerMinMax[rnn.length-1].min, rnnLayerMinMax[rnn.length-1].max])
        .range([0, nodeLength]);

  svg_rnn.selectAll('g.node-output').each(
    (d, i, g) => drawOutputScore(d, i, g, outputRectScale)
  );    
}

/**
 * create image nodes and rects for middle layers
 * 
 * @param {object} nodeGroups 'node-group' 'layer-i-node-j'
 * @param {number} left distance for x
 * @param {number} l index of current layer
 */
const initMiddleLayer = (nodeGroups, left, l) => {
   // Embed raster image in these groups
   nodeGroups.append('image')
   .attr('class', 'node-image')
   .attr('width', (d, i) => d.type!=='embedding'? nodeLength:embeddingLen)
   .attr('height', (d, i) => d.type!=='embedding'? nodeHeight:inputNodeHeight)
   .attr('x', left)
   .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y);

   // Add a rectangle to show the border
   nodeGroups.append('rect')
     .attr('class', 'bounding')
     .attr('width', (d, i) => d.type!=='embedding'? nodeLength:embeddingLen)
     .attr('height', (d, i) => d.type!=='embedding'? nodeHeight:inputNodeHeight)
     .attr('x', left)
     .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y)
     .style('fill', 'none')
     .style('stroke', 'gray')
     .style('stroke-width', 1)
     .classed('hidden', true);  
}

/**
 * Draw the canvas of image nodes
 * 
 * @param {number} l index of nn layer 
 */
const drawImageNodes = (l) => {
    let range = rnnLayerRanges[selectedScaleLevel_rnn][l];

    svg_rnn.select(`g#rnn-layer-group-${l}`)
      .selectAll('image.node-image')
      .each((d, i, g) => drawOutputRNN(d, i, g, range));
}

/**
 * Draw the overview
 * @param {number} width Width of the rnn group
 * @param {number} height Height of the rnn group
 * @param {object} rnnGroup Group to appen rnn elements to
 * @param {function} nodeMouseOverHandler Callback func for mouseOver
 * @param {function} nodeMouseLeaveHandler Callback func for mouseLeave
 * @param {function} nodeClickHandler Callback func for click
 */
export const drawRNN = (width, height, rnnGroup, nodeMouseOverHandler,
  nodeMouseLeaveHandler, nodeClickHandler) => {
 
  // Draw the RNN
  hSpaceAroundGap_rnn = calcHorSpaceGap(width);
  hSpaceAroundGapStore_rnn.set(hSpaceAroundGap_rnn);
  
  let leftAccuumulatedSpace = 0;
  // clear nodeCoordinate_rnn for reloading, otherwise it will push new coords into 
  //existing array as well as generate blank array
  nodeCoordinate_rnn.length=0;
  // Iterate through the rnn to draw nodes in each layer
  for (let l = 0; l < rnn.length; l++) {

    let curLayer = rnn[l];
    let isInput = curLayer[0].layerName === 'input';
    let isOutput = curLayer[0].layerName === 'dense_Dense1';

    nodeCoordinate_rnn.push([]);

    // Compute the x coordinate of the whole layer
    // Output (dense) and lstm layer has long gaps
    if (isOutput || curLayer[0].type ==='lstm') {
      leftAccuumulatedSpace += hSpaceAroundGap_rnn * gapRatio;
    } else {
      leftAccuumulatedSpace += hSpaceAroundGap_rnn;
    }

    // All nodes share the same x coordiante (left in div style)
    let left = leftAccuumulatedSpace;

    let curLayerName = curLayer[0].layerName;
    vSpaceAroundGap_rnn = curLayerName !== 'input' && !curLayerName.includes('embbeding')
        ? calcVerSpaceGap(height, curLayer.length, nodeHeight) 
        : calcVerSpaceGap(height, curLayer.length, inputNodeHeight);
    vSpaceAroundGapStore_rnn.set(vSpaceAroundGap_rnn);

    let layerGroup = rnnGroup.append('g')
      .attr('class', 'rnn-layer-group')
      .attr('id', `rnn-layer-group-${l}`);

    let nodeGroups = layerGroup.selectAll('g.node-group')
      .data(curLayer, d => d.index)
      .enter()
      .append('g')
      .attr('class', 'node-group')
      .style('cursor', 'pointer')
      .style('pointer-events', 'all')
      .on('click', nodeClickHandler)
      .on('mouseover', nodeMouseOverHandler)
      .on('mouseleave', nodeMouseLeaveHandler)
      .classed('node-input', isInput)
      .classed('node-output', isOutput)
      .attr('id', (d, i) => {
          // Compute the coordinate
          // Not using transform on the group object because of a decade old
          // bug on webkit (safari)
          // https://bugs.webkit.org/show_bug.cgi?id=23113
          let top;
          if (d.type !=='input'){
              top = i * nodeHeight + (i + 1) * vSpaceAroundGap_rnn;
            } else{
              top = i * inputNodeHeight + (i + 1) * vSpaceAroundGap_rnn;
            }

          top += svgPaddings.top;
          nodeCoordinate_rnn[l].push({x: left, y: top});
          return `layer-${l}-node-${i}`
        }
      );

    // Overwrite the mouseover and mouseleave function 
    // for input nodes to showhover info in the UI
    layerGroup.selectAll('g.node-input')
      .on('mouseover', (d, i, g) => {
        nodeMouseOverHandler(d, i, g);
        let word = d.output[0] === 0 ? '<pad>': reviewArray[i + reviewArray.length - curLayer.length]
        hoverInfoStore_rnn.set( {show: true, text: `'${word}' token in vocabulary is: ${d.output[0]}`} );
      })
      .on('mouseleave', (d, i, g) => {
        nodeMouseLeaveHandler(d, i, g);
        let word = d.output[0] === 0 ? '<pad>': reviewArray[i + reviewArray.length - curLayer.length]
        hoverInfoStore_rnn.set( {show: false, text: `'${word}' token in vocabulary is: ${d.output[0]}`} );
      }
    );

    // Overwrite the mouseover and mouseleave function for output nodes to show
    // hover info in the UI
    layerGroup.selectAll('g.node-output')
      .on('mouseover', (d, i, g) => {
        nodeMouseOverHandler(d, i, g);
        hoverInfoStore_rnn.set( {show: true, text: `Output value: ${formater(d.output)}`} );
      })
      .on('mouseleave', (d, i, g) => {
        nodeMouseLeaveHandler(d, i, g);
        hoverInfoStore_rnn.set( {show: false, text: `Output value: ${formater(d.output)}`} );
      }
    );
    
    if (curLayer[0].layerName === 'input') {
        initInputLayer(nodeGroups, left, l, curLayer);
        drawImageNodes(l);
      } else if (curLayer[0].layerName.includes('dense')){ 
        initOutputLayer(nodeGroups, left, l);
        drawOutPutLayer();
      } else {
        initMiddleLayer(nodeGroups, left, l);
        drawImageNodes(l);
      }
    // add the length of node
    leftAccuumulatedSpace += rnn[l][0].type !== 'embedding'? nodeLength: embeddingLen;;
  }

  // Share the nodeCoordinate
  nodeCoordinateStore_rnn.set(nodeCoordinate_rnn);

  // Add layer label
  addLabels();

  // Add layer color scale legends
  addLegends();

  // Add edges between nodes
  addEdges(rnnGroup);
}

/**
 * Update canvas values when user changes input image
 */
export const updateRNN = () => {
  console.log('reviewArray is: ', reviewArray);

  // Compute the scale of the output score width (mapping the the node
  // width to the max output score)
  let outputRectScale = d3.scaleLinear()
      .domain([rnnLayerMinMax[rnn.length-1].min, rnnLayerMinMax[rnn.length-1].max])
      .range([0, nodeLength]);

  // Rebind the rnn data to layer groups layer by layer
  for (let l = 0; l < rnn.length; l++) {
    let curLayer = rnn[l];
    let range = rnnLayerRanges[selectedScaleLevel_rnn][l];
    let layerGroup = svg_rnn.select(`g#rnn-layer-group-${l}`);

    let nodeGroups = layerGroup.selectAll('g.node-group')
      .data(curLayer);

    if (l < rnn.length - 1) {
      // Redraw the canvas and output node
      nodeGroups.transition('disappear')
        .duration(300)
        .ease(d3.easeCubicOut)
        .style('opacity', 0)
        .on('end', function() {
          d3.select(this)
            .select('image.node-image')
            .each((d, i, g) => drawOutputRNN(d, i, g, range));
          d3.select(this)
            .select('text.input-annotation')
            .text((d,i) => d.output);
          // d3.select(this)
          //   .select('text.input-text')
          //   .text((d,i,g) => d.output[0] === 0 
          //     ? `<pad>`:`${reviewArray[d.index+reviewArray.length-rnn[l].length]}` );
          d3.select(this).transition('appear')
            .duration(700)
            .ease(d3.easeCubicIn)
            .style('opacity', 1);
        });
    } else {
      nodeGroups.each(
        (d, i, g) => drawOutputScore(d, i, g, outputRectScale)
      );
    }
  }

  // Update the color scale legend
  // Local legends
  for (let i = 0; i < num_stack; i++){
    let start = 1 + i * num_module;
    let domain1 = rnnLayerRanges.local[start];
    let range1 =  1 * embeddingLen + 0 * hSpaceAroundGap_rnn;
    legendUpdate(range1, domain1,'local',i , 1);
    
    let range2 = 1 * nodeLength + 0 * hSpaceAroundGap_rnn;
    let domain2 =  rnnLayerRanges.local[start + 1];
    legendUpdate(range2, domain2,'local',i , 2)
  }

  // Module legend
  for (let i = 0; i < num_stack; i++){
    let start = 1 + i * num_module;
    // 1 embedding, 1 long gap and 1 lstm
    let range = 1 * nodeLength + 1 * embeddingLen + 0 * hSpaceAroundGap_rnn +
          1 * hSpaceAroundGap_rnn * gapRatio - 1.2;
    let domain = rnnLayerRanges.module[start];
    legendUpdate(range,domain,'module',i);
  }

  // Global legend
  let start = 1;
  // evething but input node, output node and 2 long gap
  let range = (numLayers-3) * nodeLength + 1 * embeddingLen + (n_shortGaps-3) * hSpaceAroundGap_rnn 
           + (n_longGaps-1) * hSpaceAroundGap_rnn * gapRatio - 1.2;
  let domain = rnnLayerRanges.global[start];
  legendUpdate(range, domain, 'global');
}

/**
 * Update the ranges for current CNN layers
 */
export const updateRNNLayerRanges = (maxLen =1) => {
  // Iterate through all nodes to find a output ranges for each layer
  let rnnLayerRangesLocal = [maxLen];
  // let rnnLayerRangesLocal = [1];
  let curRange = undefined;

  // Also track the min/max of each layer (avoid computing during intermediate
  // layer)
  rnnLayerMinMax = [];

  for (let l = 0; l < rnn.length - 1; l++) {
    let curLayer = rnn[l];

    // Compute the min max
    let outputExtents = curLayer.map(d => getExtent(d.output));
    let aggregatedExtent = outputExtents.reduce((acc, cur) => {
      return [Math.min(acc[0], cur[0]), Math.max(acc[1], cur[1])];
    })

    // if (l === 0)  { 
    //   console.log(outputExtents,aggregatedExtent);
    // }

    rnnLayerMinMax.push({min: aggregatedExtent[0], max: aggregatedExtent[1]});

    // input layer refreshes curRange counting to [0, 1]
    // because there are too many words in dictionary
    // if (curLayer[0].type ==='input') {
    //   aggregatedExtent = aggregatedExtent.map(i => i/aggregatedExtent[1]);
    //   // console.log('divided by input dims');
    //   // Plus 0.1 to offset the rounding error (avoid black color)
    //   curRange = 2 * (0.1 + 
    //     Math.round(Math.max(...aggregatedExtent) * 1000) / 1000);
    // } 
    // embedding and lstm layers refresh curRange counting
    if (curLayer[0].type === 'embedding' || curLayer[0].type === 'lstm') {
      aggregatedExtent = aggregatedExtent.map(Math.abs);
      // Plus 0.1 to offset the rounding error (avoid black color)
      curRange = 2 * (0.1 + 
        Math.round(Math.max(...aggregatedExtent) * 1000) / 1000);
    }

    if (curRange !== undefined){
      rnnLayerRangesLocal.push(curRange);
    }
  }

  // Finally, add the output layer range
  rnnLayerRangesLocal.push(1);
  rnnLayerMinMax.push({min: 0, max: 1});

  // Support different levels of scales (1) lcoal, (2) component, (3) global
  let rnnLayerRangesComponent = [maxLen];
  let numOfComponent = (numLayers - 2) / num_module;
  for (let i = 0; i < numOfComponent; i++) {
    let curArray = rnnLayerRangesLocal.slice(1 + num_module * i, 1 + num_module * i + num_module);
    let maxRange = Math.max(...curArray);
    for (let j = 0; j < num_module; j++) {
      rnnLayerRangesComponent.push(maxRange);
    }
  }
  rnnLayerRangesComponent.push(1);

  let rnnLayerRangesGlobal = [maxLen];
  let maxRange = Math.max(...rnnLayerRangesLocal.
                  slice(1, rnnLayerRangesLocal.length - 1));
  for (let i = 0; i < numLayers - 2; i++) {
    rnnLayerRangesGlobal.push(maxRange);
  }
  rnnLayerRangesGlobal.push(1);

  // Update the ranges dictionary
  rnnLayerRanges.local = rnnLayerRangesLocal;
  rnnLayerRanges.module = rnnLayerRangesComponent;
  rnnLayerRanges.global = rnnLayerRangesGlobal;
  rnnLayerRanges.output = [0, d3.max(rnn[rnn.length - 1].map(d => d.output))];

  rnnLayerRangesStore.set(rnnLayerRanges);
  rnnLayerMinMaxStore.set(rnnLayerMinMax);
}