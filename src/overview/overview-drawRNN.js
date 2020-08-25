/* global d3, SmoothScroll */

import { 
  svgStore_rnn, vSpaceAroundGapStore_rnn, hSpaceAroundGapStore_rnn, rnnStore,
  nodeCoordinateStore_rnn, selectedScaleLevelStore_rnn, rnnLayerRangesStore,
  detailedModeStore_rnn, rnnLayerMinMaxStore, hoverInfoStore_rnn
} from '../stores.js';
import {
  getExtent, getLinkDataRNN
} from './draw-utils.js';
import { rnnOverviewConfig, overviewConfig } from '../config.js';

// Configs
const layerColorScales = rnnOverviewConfig.layerColorScales;
const nodeLength = rnnOverviewConfig.nodeLength;
const nodeHeight = rnnOverviewConfig.nodeHeight;
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

// There are 0 short gaps and 3 long gaps
let n_shortGaps = 2;
let n_longGaps = 3;
// for module view
let num_module = 2;
let num_stack = 1;

/**
 * Use bounded d3 data to draw one canvas
 * @param {object} d d3 data
 * @param {index} i d3 data index
 * @param {[object]} g d3 group
 * @param {number} range color range map (max - min)
 */
export const drawOutputRNN = (d, i, g, range) => {
  let image = g[i];
  let colorScale = layerColorScales[d.type];

  // if (d.type === 'input') {
  //   colorScale = colorScale[d.index];
  // }

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

  if (imageLength === 1) {
    // imageSingleArray[0] = d.output;
    // for output is number, for instance, LSTM..
    let color = d3.rgb(colorScale((d.output+range/2)/range));
    imageSingleArray[0] = color.r;
    imageSingleArray[1] = color.g;
    imageSingleArray[2] = color.b;
    imageSingleArray[3] = 255;
  } else if (d.type.includes('embedding')){
    for (let j = 0; j < imageLength; j++){
      let color = d3.rgb(colorScale((d.output[j] + range/2)/range));

      imageSingleArray[4*j] = color.r;
      imageSingleArray[4*j + 1] = color.g;
      imageSingleArray[4*j + 2] = color.b;
      imageSingleArray[4*j + 3] = 255;
    }
  }else{
    for (let i = 0; i < imageSingleArray.length; i+=4) {
      let pixeIndex = Math.floor(i / 4);
      let row = Math.floor(pixeIndex / imageLength);
      let column = pixeIndex % imageLength;
      let color = undefined;
      if (d.type === 'input' || d.type === 'fc' ) {
        color = d3.rgb(colorScale(1 - d.output[row][column]));
      } else {
        color = d3.rgb(colorScale((d.output[row][column] + range / 2) / range));
      }

      imageSingleArray[i] = color.r;
      imageSingleArray[i + 1] = color.g;
      imageSingleArray[i + 2] = color.b;
      imageSingleArray[i + 3] = 255;
    }
  }
  

  if(i === 127) {
    console.log(imageSingle);
  }

  // canvas.toDataURL() only exports image in 96 DPI, so we can hack it to have
  // higher DPI by rescaling the image using canvas magic
  let largeCanvas = document.createElement('canvas');
  largeCanvas.width = nodeLength*3 ;
  largeCanvas.height = nodeHeight*3 ;
  let largeCanvasContext = largeCanvas.getContext('2d');

  // Use drawImage to resize the original pixel array, and put the new image
  // (canvas) into corresponding canvas
  bufferContext.putImageData(imageSingle, 0, 0);
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
  largeCanvasContext.drawImage(bufferCanvas, 0, 0, imageLength, imageHeight,
      0, 0, nodeLength*3, nodeHeight*3);

  
  let imageDataURL = largeCanvas.toDataURL();
  d3.select(image).attr('xlink:href', imageDataURL);

  // Destory the buffer canvas
  bufferCanvas.remove();
  largeCanvas.remove();
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
  group.select('rect.output-rect')
    .transition('dense')
    .delay(500)
    .duration(800)
    .ease(d3.easeCubicIn)
    .attr('width', scale(d.output))
    .attr('fill', scale(d.output[1]));   
}

export const drawCustomReivew = (image, inputLayer) => {

  let imageWidth = image.width;
  // Set up a second convas in order to resize image
  let imageLength = inputLayer[0].output.length;
  let bufferCanvas = document.createElement("canvas");
  let bufferContext = bufferCanvas.getContext("2d");
  bufferCanvas.width = imageLength;
  bufferCanvas.height = imageLength;

  // Fill image pixel array
  let imageSingle = bufferContext.getImageData(0, 0, imageLength, imageLength);
  let imageSingleArray = imageSingle.data;

  for (let i = 0; i < imageSingleArray.length; i+=4) {
    let pixeIndex = Math.floor(i / 4);
    let row = Math.floor(pixeIndex / imageLength);
    let column = pixeIndex % imageLength;

    let red = inputLayer[0].output[row][column];
    let green = inputLayer[1].output[row][column];
    let blue = inputLayer[2].output[row][column];

    imageSingleArray[i] = red * 255;
    imageSingleArray[i + 1] = green * 255;
    imageSingleArray[i + 2] = blue * 255;
    imageSingleArray[i + 3] = 255;
  }

  // canvas.toDataURL() only exports image in 96 DPI, so we can hack it to have
  // higher DPI by rescaling the image using canvas magic
  let largeCanvas = document.createElement('canvas');
  largeCanvas.width = imageWidth * 3;
  largeCanvas.height = imageWidth * 3;
  let largeCanvasContext = largeCanvas.getContext('2d');

  // Use drawImage to resize the original pixel array, and put the new image
  // (canvas) into corresponding canvas
  bufferContext.putImageData(imageSingle, 0, 0);
  largeCanvasContext.drawImage(bufferCanvas, 0, 0, imageLength, imageLength,
    0, 0, imageWidth * 3, imageWidth * 3);
  
  let imageDataURL = largeCanvas.toDataURL();
  // d3.select(image).attr('xlink:href', imageDataURL);
  image.src = imageDataURL;

  // Destory the buffer canvas
  bufferCanvas.remove();
  largeCanvas.remove();
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

/**
 * Draw all legends
 * @param {object} legends Parent group
 * @param {number} legendHeight Height of the legend element
 */
const drawLegends = (legends, legendHeight) => {
  // Add local legends
  for (let i = 0; i < num_stack; i++){
    let start = 1 + i * num_module;
    let range1 = rnnLayerRanges.local[start];
    let range2 = rnnLayerRanges.local[start + 1];

    let localLegendScale1 = d3.scaleLinear()
      .range([0, (num_module -1) * nodeLength 
        + (num_module-2) * hSpaceAroundGap_rnn- 1.2])
      .domain([-range1 / 2, range1 / 2]);
    
    let localLegendScale2 = d3.scaleLinear()
      .range([0, (num_module -1) * nodeLength 
        + (num_module-2) * hSpaceAroundGap_rnn - 1.2])
      .domain([-range2 / 2, range2 / 2]);

    let localLegendAxis1 = d3.axisBottom()
      .scale(localLegendScale1)
      .tickFormat(d3.format('.1f'))
      .tickValues([-range1 / 2, 0, range1 / 2]);
    
    let localLegendAxis2 = d3.axisBottom()
      .scale(localLegendScale2)
      .tickFormat(d3.format('.1f'))
      .tickValues([-range2 / 2, 0, range2 / 2]);

    let localLegend1 = legends.append('g')
      .attr('class', 'legend local-legend')
      .attr('id', `local-legend-${i}-1`)
      .classed('hidden', !detailedMode_rnn || selectedScaleLevel_rnn !== 'local')
      .attr('transform', `translate(${nodeCoordinate_rnn[start][0].x}, ${0})`);

    localLegend1.append('g')
      .attr('transform', `translate(0, ${legendHeight - 3})`)
      .call(localLegendAxis1)

    localLegend1.append('rect')
      .attr('width', (num_module -1) * nodeLength + (num_module-2) * hSpaceAroundGap_rnn)
      .attr('height', legendHeight)
      .style('fill', 'url(#embeddingGradient)');

    let localLegend2 = legends.append('g')
      .attr('class', 'legend local-legend')
      .attr('id', `local-legend-${i}-2`)
      .classed('hidden', !detailedMode_rnn || selectedScaleLevel_rnn !== 'local')
      .attr('transform', `translate(${nodeCoordinate_rnn[start + 1][0].x}, ${0})`);

    localLegend2.append('g')
      .attr('transform', `translate(0, ${legendHeight - 3})`)
      .call(localLegendAxis2)

    localLegend2.append('rect')
      .attr('width', (num_module -1) * nodeLength + (num_module-2) * hSpaceAroundGap_rnn)
      .attr('height', legendHeight)
      .style('fill', 'url(#lstmGradient)');
  }

  // Add module legends
  for (let i = 0; i < num_stack; i++){
    let start = 1 + i * num_module;
    let range = rnnLayerRanges.module[start];

    let moduleLegendScale = d3.scaleLinear()
      .range([0, num_module * nodeLength + (num_module-2) * hSpaceAroundGap_rnn +
        (num_module-1) * hSpaceAroundGap_rnn * gapRatio - 1.2])
      .domain([-range / 2, range / 2]);

    let moduleLegendAxis = d3.axisBottom()
      .scale(moduleLegendScale)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range / 2, -(range / 4), 0, range / 4, range / 2]);

    let moduleLegend = legends.append('g')
      .attr('class', 'legend module-legend')
      .attr('id', `module-legend-${i}`)
      .classed('hidden', !detailedMode_rnn || selectedScaleLevel_rnn !== 'module')
      .attr('transform', `translate(${nodeCoordinate_rnn[start][0].x}, ${0})`);
    
    moduleLegend.append('g')
      .attr('transform', `translate(0, ${legendHeight - 3})`)
      .call(moduleLegendAxis)

    moduleLegend.append('rect')
      .attr('width', num_module * nodeLength + (num_module-2) * hSpaceAroundGap_rnn +
        (num_module-1) * hSpaceAroundGap_rnn * gapRatio)
      .attr('height', legendHeight)
      .style('fill', 'url(#embeddingGradient)');
  }

  // Add global legends
  let start = 1;
  let range = rnnLayerRanges.global[start];

  let globalLegendScale = d3.scaleLinear()
    .range([0, (numLayers-2) * nodeLength
      + (n_shortGaps-2) * hSpaceAroundGap_rnn +
     (n_longGaps-2) * hSpaceAroundGap_rnn * gapRatio - 1.2])
    .domain([-range / 2, range / 2]);

  let globalLegendAxis = d3.axisBottom()
    .scale(globalLegendScale)
    .tickFormat(d3.format('.2f'))
    .tickValues([-range / 2, -(range / 4), 0, range / 4, range / 2]);

  let globalLegend = legends.append('g')
    .attr('class', 'legend global-legend')
    .attr('id', 'global-legend')
    .classed('hidden', !detailedMode_rnn || selectedScaleLevel_rnn !== 'global')
    .attr('transform', `translate(${nodeCoordinate_rnn[start][0].x}, ${0})`);

  globalLegend.append('g')
    .attr('transform', `translate(0, ${legendHeight - 3})`)
    .call(globalLegendAxis);

  globalLegend.append('rect')
    .attr('width', (numLayers-2) * nodeLength
       + (n_shortGaps-2) * hSpaceAroundGap_rnn +
      (n_longGaps-2) * hSpaceAroundGap_rnn * gapRatio)
    .attr('height', legendHeight)
    .style('fill', 'url(#embeddingGradient)');


  // Add output legend
  let outputRectScale = d3.scaleLinear()
        .domain([rnnLayerMinMax[nodeCoordinate_rnn.length-1].min, 
          rnnLayerMinMax[nodeCoordinate_rnn.length-1].max])
        .range([0, nodeLength - 1.2]);

  let outputLegendAxis = d3.axisBottom()
    .scale(outputRectScale)
    .tickFormat(d3.format('.1f'))
    .tickValues([0, rnnLayerMinMax[nodeCoordinate_rnn.length-1].max/2, 
      rnnLayerMinMax[nodeCoordinate_rnn.length-1].max])
  
  let outputLegend = legends.append('g')
    .attr('class', 'legend output-legend')
    .attr('id', 'output-legend')
    .classed('hidden', !detailedMode_rnn)
    .attr('transform', `translate(${nodeCoordinate_rnn[nodeCoordinate_rnn.length-1][0].x}, ${0})`);
  
  outputLegend.append('g')
    .attr('transform', `translate(0, ${legendHeight - 3})`)
    .call(outputLegendAxis);

  outputLegend.append('rect')
    .attr('x', 0.3)
    .attr('width', nodeLength - 0.3)
    .attr('height', legendHeight)
    .attr('transform', `rotate(180, ${nodeLength/2}, ${legendHeight/2})`)
    .style('stroke', 'rgb(20, 20, 20)')
    .style('stroke-width', 0.3)
    .style('fill', 'url(#denseGradient)');
  
  // Add input image legend
  let inputScale = d3.scaleLinear()
    .range([0, nodeLength - 1.2])
    .domain([0, 1]);

  let inputLegendAxis = d3.axisBottom()
    .scale(inputScale)
    .tickFormat(d3.format('.1f'))
    .tickValues([0, 0.5, 1]);

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
    .attr('transform', `rotate(180, ${nodeLength/2}, ${legendHeight/2})`)
    .style('stroke', 'rgb(20, 20, 20)')
    .style('stroke-width', 0.3)
    .style('fill', 'url(#inputGradient)');
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
  nodeMouseLeaveHandler, nodeClickHandler, inputTextList) => {
  // Draw the RNN

  hSpaceAroundGap_rnn = (width - nodeLength * numLayers) 
    / (n_shortGaps + n_longGaps * gapRatio);
  // console.log('horizontal space around gap is: ', hSpaceAroundGap_rnn);
  hSpaceAroundGapStore_rnn.set(hSpaceAroundGap_rnn);
  let leftAccuumulatedSpace = 0;

  // Iterate through the rnn to draw nodes in each layer
  for (let l = 0; l < rnn.length; l++) {

    let curLayer = rnn[l];
    let isOutput = curLayer[0].layerName === 'dense_Dense1';

    nodeCoordinate_rnn.push([]);

    // Compute the x coordinate of the whole layer
    // Output layer and conv layer has long gaps
    if (isOutput || curLayer[0].type === 'embedding' || curLayer[0].type ==='lstm') {
      leftAccuumulatedSpace += hSpaceAroundGap_rnn * gapRatio;
    } else {
      leftAccuumulatedSpace += hSpaceAroundGap_rnn;
    }

    // All nodes share the same x coordiante (left in div style)
    let left = leftAccuumulatedSpace;
    // let meaningfulInputLen;

    let layerGroup = rnnGroup.append('g')
      .attr('class', 'rnn-layer-group')
      .attr('id', `rnn-layer-group-${l}`);

    // // igonre the count of y when the node contains <pad> in the input layer
    // if (curLayer[0].layerName === 'input') {
    //   meaningfulInputLen = curLayer.filter(d =>d.output !== 0).length;
    //   vSpaceAroundGap_rnn = (height - nodeHeight * meaningfulInputLen) /
    //   (meaningfulInputLen + 1);
    // } else {
      vSpaceAroundGap_rnn = (height - nodeHeight * curLayer.length) /
      (curLayer.length + 1);
    // }

    vSpaceAroundGapStore_rnn.set(vSpaceAroundGap_rnn);

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
      .classed('node-output', isOutput)
      .attr('id', (d, i) => {
        // Compute the coordinate
        // Not using transform on the group object because of a decade old
        // bug on webkit (safari)
        // https://bugs.webkit.org/show_bug.cgi?id=23113
        let top = i * nodeHeight + (i + 1) * vSpaceAroundGap_rnn;
        top += svgPaddings.top;
        nodeCoordinate_rnn[l].push({x: left, y: top});
        return `layer-${l}-node-${i}`
      });
    
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
      });

    // let pixelWidth = nodeLength / 2;
    // let totalLength = (2 * nodeLength +
    //   5.5 * hSpaceAroundGap * gapRatio + pixelWidth);
    // let leftX = nodeCoordinate_rnn[l+1][0].x - totalLength;
    // let intermediateGap = (hSpaceAroundGap * gapRatio * 4) / 2;

    // let intermediateX1 = leftX + nodeLength + intermediateGap;
    // let intermediateX2 = intermediateX1 + intermediateGap + pixelWidth;

    let range = rnnLayerRanges[selectedScaleLevel_rnn][l];
    // ??? d.type or conv
    let colorScale = layerColorScales[curLayer[0].type];
    // ??? divided bb conv units???
    let extractedLength = rnn[l].length / 1

    let topY = nodeCoordinate_rnn[l][0].y;
    let bottomY = nodeCoordinate_rnn[l][rnn[l].length-1].y + nodeLength -
      extractedLength * nodeHeight;

    
    if (curLayer[0].layerName === 'input') {
      // // Embed raster image in these groups
      // nodeGroups.append('image')
      //   .attr('class', 'node-image')
      //   .attr('width', nodeLength)
      //   .attr('height', nodeLength)
      //   .attr('x', left)
      //   .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y);      

      nodeGroups.append('rect')
        .attr('class', 'input-rect')
        .attr('width', nodeLength)
        .attr('height', nodeHeight)
        .attr('x', left)
        .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y)
        .style('cursor', 'crosshair')
        .style('fill', (d, i) => colorScale((curLayer[i].output+range/2)/range))
        .style('stroke', 'gray')
        .style('stroke-width', 0.1)
      nodeGroups.append('text')
        .attr('class', 'input-text')
        .attr('x', svgPaddings.left/3)
        .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y)
        .style('dominant-baseline', 'middle')
        .style('font-size', '8px')
        .style('fill', 'black')
        .style('opacity', 0.8)
        .text((d, i) => inputTextList[i] === undefined? `<pad>:${d.output}`:`${inputTextList[i]}:${d.output}`);
    } else if (curLayer[0].layerName.includes('dense')){ 
      // Add a rectangle to show the border
      nodeGroups.append('rect')
      .attr('class', 'bounding')
      .attr('x', left)
      .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y + nodeLength / 2)
      .attr('height', nodeLength / 4)
      .attr('width', nodeLength)
      .style('fill', 'none')
      .style('stroke', 'gray')
      .style('stroke-width', 0.5)
      .classed('hidden', false);
      nodeGroups.append('rect')
        .attr('class', 'output-rect')
        .attr('x', left)
        .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y + nodeLength / 2)
        .attr('height', nodeLength / 4)
        .attr('width', nodeLength)
        .style('fill', 'gray');
      nodeGroups.append('text')
        .attr('class', 'output-text-score')
        .attr('x', left + nodeLength/4)
        .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y+ nodeLength)
        .style('dominant-baseline', 'middle')
        .style('font-size', '11px')
        .style('fill', 'black')
        .style('opacity', 0.5)
        .text((d, i) => d.output.toFixed(2));
      nodeGroups.append('text')
        .attr('class', 'output-text-one')
        .attr('x', left-nodeLength/4)
        .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y + nodeLength / 4)
        .style('dominant-baseline', 'middle')
        .style('font-size', '11px')
        .style('fill', 'black')
        .style('opacity', 0.5)
        .text((d, i) => classLists[i]);
      nodeGroups.append('text')
        .attr('class', 'output-text-two')
        .attr('x', left+nodeLength*3/4)
        .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y + nodeLength / 4)
        .style('dominant-baseline', 'middle')
        .style('font-size', '11px')
        .style('fill', 'black')
        .style('opacity', 0.5)
        .text((d, i) => classLists[i+1]);
      
      // Add annotation text to tell readers the exact output probability
      // nodeGroups.append('text')
      //   .attr('class', 'annotation-text')
      //   .attr('id', (d, i) => `output-prob-${i}`)
      //   .attr('x', left)
      //   .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y + 10)
      //   .text(d => `(${d3.format('.4f')(d.output)})`);
    } else {
            // Embed raster image in these groups
            nodeGroups.append('image')
            .attr('class', 'node-image')
            .attr('width', nodeLength)
            .attr('height', nodeHeight)
            .attr('x', left)
            .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y);

          // Add a rectangle to show the border
          nodeGroups.append('rect')
            .attr('class', 'bounding')
            .attr('width', nodeLength)
            .attr('height', nodeHeight)
            .attr('x', left)
            .attr('y', (d, i) => nodeCoordinate_rnn[l][i].y)
            .style('fill', 'none')
            .style('stroke', 'gray')
            .style('stroke-width', 1)
            .classed('hidden', true);    
    }

    leftAccuumulatedSpace += nodeLength;
  }

  // Share the nodeCoordinate
  nodeCoordinateStore_rnn.set(nodeCoordinate_rnn)

  // Compute the scale of the output score width (mapping the the node
  // width to the max output score)
  let outputRectScale = d3.scaleLinear()
        .domain([rnnLayerMinMax[rnn.length-1].min, rnnLayerMinMax[rnn.length-1].max])
        .range([0, nodeLength]);

  // Draw the canvas
  for (let l = 0; l < rnn.length; l++) {
    let range = rnnLayerRanges[selectedScaleLevel_rnn][l];

    svg_rnn.select(`g#rnn-layer-group-${l}`)
      .selectAll('image.node-image')
      .each((d, i, g) => drawOutputRNN(d, i, g, range));
  }

  svg_rnn.selectAll('g.node-output').each(
    (d, i, g) => drawOutputScore(d, i, g, outputRectScale)
  );

  // Add layer label
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
      let x = nodeCoordinate_rnn[i][0].x + nodeLength / 2;
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
      let x = nodeCoordinate_rnn[i][0].x + nodeLength / 2;
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

  // Add layer color scale legends
  getLegendGradient(svg_rnn, layerColorScales.embedding, 'embeddingGradient');
  getLegendGradient(svg_rnn, layerColorScales.lstm, 'lstmGradient');
  getLegendGradient(svg_rnn, layerColorScales.input, 'inputGradient');
  getLegendGradient(svg_rnn, layerColorScales.dense, 'denseGradient');

  let legendHeight = 5;
  // y is based on the last one of vSpaceAroundGap_rnn
  let legends = svg_rnn.append('g')
      .attr('class', 'color-legend')
      .attr('transform', `translate(${0}, ${
        svgPaddings.top + vSpaceAroundGap_rnn 
          * (rnn[rnn.length-1].length+1) + nodeLength / 5
      })`);
  
  drawLegends(legends, legendHeight);

  // Add edges between nodes
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);
  
  let linkData = getLinkDataRNN(nodeCoordinate_rnn, rnn);
  console.log('linkData is: ', linkData);
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
    .style('stroke-width', d =>d.targetLayerIndex !==3 ? edgeStrokeWidth:edgeStrokeWidth*3)
    .style('opacity', edgeOpacity)
    .style('stroke', edgeInitColor);

  // // Add input channel annotations
  // let inputAnnotation = rnnGroup.append('g')
  //   .attr('class', 'input-annotation');

  // let redChannel = inputAnnotation.append('text')
  //   .attr('x', nodeCoordinate_rnn[0][0].x + nodeLength / 2)
  //   .attr('y', nodeCoordinate_rnn[0][0].y + nodeLength + 5)
  //   .attr('class', 'annotation-text')
  //   .style('dominant-baseline', 'hanging')
  //   .style('text-anchor', 'middle');
  
  // redChannel.append('tspan')
  //   .style('dominant-baseline', 'hanging')
  //   .style('fill', '#C95E67')
  //   .text('Red');
  
  // redChannel.append('tspan')
  //   .style('dominant-baseline', 'hanging')
  //   .text(' channel');

  // inputAnnotation.append('text')
  //   .attr('x', nodeCoordinate_rnn[0][1].x + nodeLength / 2)
  //   .attr('y', nodeCoordinate_rnn[0][1].y + nodeLength + 5)
  //   .attr('class', 'annotation-text')
  //   .style('dominant-baseline', 'hanging')
  //   .style('text-anchor', 'middle')
  //   .style('fill', '#3DB665')
  //   .text('Green');

  // inputAnnotation.append('text')
  //   .attr('x', nodeCoordinate_rnn[0][2].x + nodeLength / 2)
  //   .attr('y', nodeCoordinate_rnn[0][2].y + nodeLength + 5)
  //   .attr('class', 'annotation-text')
  //   .style('dominant-baseline', 'hanging')
  //   .style('text-anchor', 'middle')
  //   .style('fill', '#3F7FBC')
  //   .text('Blue');
}

/**
 * Update canvas values when user changes input image
 */
export const updateRNN = () => {
  // Compute the scale of the output score width (mapping the the node
  // width to the max output score)
  let outputRectScale = d3.scaleLinear()
      .domain(rnnLayerRanges.output)
      .range([0, nodeLength]);

  // Rebind the cnn data to layer groups layer by layer
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
    let range1 = rnnLayerRanges.local[start];
    let range2 = rnnLayerRanges.local[start + 2];

    let localLegendScale1 = d3.scaleLinear()
      .range([0, 2 * nodeLength + hSpaceAroundGap_rnn])
      .domain([-range1, range1]);
    
    let localLegendScale2 = d3.scaleLinear()
      .range([0, 3 * nodeLength + 2 * hSpaceAroundGap_rnn])
      .domain([-range2, range2]);

    let localLegendAxis1 = d3.axisBottom()
      .scale(localLegendScale1)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range1, 0, range1]);
    
    let localLegendAxis2 = d3.axisBottom()
      .scale(localLegendScale2)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range2, 0, range2]);
    
    svg_rnn.select(`g#local-legend-${i}-1`).select('g').call(localLegendAxis1);
    svg_rnn.select(`g#local-legend-${i}-2`).select('g').call(localLegendAxis2);
  }

  // Module legend
  for (let i = 0; i < num_stack; i++){
    let start = 1 + i * num_module;
    let range = rnnLayerRanges.local[start];

    let moduleLegendScale = d3.scaleLinear()
      .range([0, num_module * nodeLength + 3 * hSpaceAroundGap_rnn +
        1 * hSpaceAroundGap_rnn * gapRatio - 1.2])
      .domain([-range, range]);

    let moduleLegendAxis = d3.axisBottom()
      .scale(moduleLegendScale)
      .tickFormat(d3.format('.2f'))
      .tickValues([-range, -(range / 2), 0, range/2, range]);
    
    svg_rnn.select(`g#module-legend-${i}`).select('g').call(moduleLegendAxis);
  }

  // Global legend
  let start = 1;
  let range = rnnLayerRanges.global[start];

  let globalLegendScale = d3.scaleLinear()
    .range([0, 10 * nodeLength + 6 * hSpaceAroundGap_rnn +
      3 * hSpaceAroundGap_rnn * gapRatio - 1.2])
    .domain([-range, range]);

  let globalLegendAxis = d3.axisBottom()
    .scale(globalLegendScale)
    .tickFormat(d3.format('.2f'))
    .tickValues([-range, -(range / 2), 0, range/2, range]);

  svg_rnn.select(`g#global-legend`).select('g').call(globalLegendAxis);

  // Output legend
  let outputLegendAxis = d3.axisBottom()
    .scale(outputRectScale)
    .tickFormat(d3.format('.1f'))
    .tickValues([0, rnnLayerMinMax[rnn.length-1].max]);
  
  svg_rnn.select('g#output-legend').select('g').call(outputLegendAxis);
}

/**
 * Update the ranges for current CNN layers
 */
export const updateRNNLayerRanges = (inputDim=1) => {
  // Iterate through all nodes to find a output ranges for each layer
  let rnnLayerRangesLocal = [1];
  let curRange = undefined;

  // Also track the min/max of each layer (avoid computing during intermediate
  // layer)
  rnnLayerMinMax = [];

  for (let l = 0; l < rnn.length - 1; l++) {
    let curLayer = rnn[l];

    // Compute the min max
    let outputExtents = curLayer.map(l => getExtent(l.output));
    let aggregatedExtent = outputExtents.reduce((acc, cur) => {
      return [Math.min(acc[0], cur[0]), Math.max(acc[1], cur[1])];
    })

    // input layer refreshes curRange counting to [0, 1]
    // because there are too many words in dictionary
    if (curLayer[0].type ==='input') {
      aggregatedExtent = aggregatedExtent.map(i => i/aggregatedExtent[1]);
      // console.log('divided by input dims');
    } 
    rnnLayerMinMax.push({min: aggregatedExtent[0], max: aggregatedExtent[1]});

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
  let rnnLayerRangesComponent = [1];
  let numOfComponent = (numLayers - 2) / num_module;
  for (let i = 0; i < numOfComponent; i++) {
    let curArray = rnnLayerRangesLocal.slice(1 + num_module * i, 1 + num_module * i + num_module);
    let maxRange = Math.max(...curArray);
    for (let j = 0; j < num_module; j++) {
      rnnLayerRangesComponent.push(maxRange);
    }
  }
  rnnLayerRangesComponent.push(1);

  let rnnLayerRangesGlobal = [1];
  let maxRange = Math.max(...rnnLayerRangesLocal.slice(1,
    rnnLayerRangesLocal.length - 1));
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