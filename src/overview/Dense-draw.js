/* global d3, SmoothScroll */

import {
  svgStore_rnn, vSpaceAroundGapStore_rnn, hSpaceAroundGapStore_rnn, rnnStore,
  nodeCoordinateStore_rnn, selectedScaleLevelStore_rnn, rnnLayerRangesStore,
  rnnLayerMinMaxStore, isInSigmoidStore_rnn, sigmoidDetailViewStore_rnn,
  hoverInfoStore_rnn, allowsSigmoidAnimationStore_rnn, detailedModeStore_rnn,
  shouldIntermediateAnimateStore_rnn
} from '../stores.js';
import {
  getOutputKnot, getInputKnot, gappedColorScale, getMidCoords
} from './drawRNN-utils.js';
import {
  drawIntermediateLayerLegend, moveLayerX, addOverlayGradient,
  drawArrow
} from './intermediateRNN-utils.js';
import { rnnOverviewConfig } from '../config.js';

// Configs
const layerColorScales = rnnOverviewConfig.layerColorScales;
const numLayer=rnnOverviewConfig.numLayers;
const nodeLength = rnnOverviewConfig.nodeLength;
const nodeHeight = rnnOverviewConfig.nodeHeight;
const inputNodeHeight = rnnOverviewConfig.inputNodeHeight;
const embeedingLen = rnnOverviewConfig.embedddingLength;
const plusSymbolRadius = rnnOverviewConfig.plusSymbolRadius;
const intermediateColor = rnnOverviewConfig.intermediateColor;
const kernelRectLength = rnnOverviewConfig.kernelRectLength;
const svgPaddings = rnnOverviewConfig.svgPaddings;
const gapRatio = rnnOverviewConfig.gapRatio;
const classList = rnnOverviewConfig.classLists;
const formater = d3.format('.4f');

// Shared variables
let svg = undefined;
svgStore_rnn.subscribe( value => {svg = value;} )

let vSpaceAroundGap = undefined;
vSpaceAroundGapStore_rnn.subscribe( value => {vSpaceAroundGap = value;} )

let hSpaceAroundGap = undefined;
hSpaceAroundGapStore_rnn.subscribe( value => {hSpaceAroundGap = value;} )

let rnn = undefined;
rnnStore.subscribe( value => {rnn = value;} )

let nodeCoordinate = undefined;
nodeCoordinateStore_rnn.subscribe( value => {nodeCoordinate = value;} )

let selectedScaleLevel = undefined;
selectedScaleLevelStore_rnn.subscribe( value => {selectedScaleLevel = value;} )

let rnnLayerRanges = undefined;
rnnLayerRangesStore.subscribe( value => {rnnLayerRanges = value;} )

let rnnLayerMinMax = undefined;
rnnLayerMinMaxStore.subscribe( value => {rnnLayerMinMax = value;} )

let isInSigmoid = undefined;
isInSigmoidStore_rnn.subscribe( value => {isInSigmoid = value;} )

let allowsSigmoidAnimation = undefined;
allowsSigmoidAnimationStore_rnn.subscribe( value => {allowsSigmoidAnimation = value;} )

let sigmoidDetailViewInfo = undefined;
sigmoidDetailViewStore_rnn.subscribe( value => {sigmoidDetailViewInfo = value;} )

let hoverInfo = undefined;
hoverInfoStore_rnn.subscribe( value => {hoverInfo = value;} )

let detailedMode = undefined;
detailedModeStore_rnn.subscribe( value => {detailedMode = value;} )

let shouldIntermediateAnimate = undefined;
shouldIntermediateAnimateStore_rnn.subscribe(value => {
  shouldIntermediateAnimate = value;
})

const layerIndexDict = {
  'input': 0,
  'embedding_Embedding1': 1,
  'lstm_LSTM1': 2,
  'dense_Dense1': 3
}

let legendY;
let labelY = svgPaddings.top / 2 - 4;
let legendHeight =5;
let hasInitialized = false;
let logits = [];
let lstmFactoredFDict = {};

const animateEdge = (d, i, g, dashoffset) => {
  let curPath = d3.select(g[i]);
  curPath.transition()
    .duration(60000)
    .ease(d3.easeLinear)
    .attr('stroke-dashoffset', dashoffset)
    .on('end', (d, i, g) => {
      if (shouldIntermediateAnimate) {
        animateEdge(d, i, g, dashoffset - 2000);
      }
    });
}

const moveLegend = (d, i, g, moveX, duration, restore) => {
  let legend = d3.select(g[i]);

  if (!restore) {
    let previousTransform = legend.attr('transform');
    let previousLegendX = +previousTransform.replace(/.*\(([\d\.]+),.*/, '$1');
    let previousLegendY = +previousTransform.replace(/.*,\s([\d\.]+)\)/, '$1');
  
    legend.transition('sigmoid')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', `translate(${previousLegendX - moveX}, ${previousLegendY})`);
    
    // If not in restore mode, we register the previous location to the DOM element
    legend.attr('data-preX', previousLegendX);
    legend.attr('data-preY', previousLegendY);
  } else {
    // Restore the recorded location
    let previousLegendX = +legend.attr('data-preX');
    let previousLegendY = +legend.attr('data-preY');

    legend.transition('sigmoid')
      .duration(duration)
      .ease(d3.easeCubicInOut)
      .attr('transform', `translate(${previousLegendX}, ${previousLegendY})`);
  }
}

const logitCircleMouseOverHandler = (i) => {
  // Update the hover info UI
  hoverInfoStore_rnn.set({
    show: true,
    text: `Logit: ${formater(logits[i])}`
  })

  // Highlight the text in the detail view
  sigmoidDetailViewInfo.highlightI = i;
  sigmoidDetailViewStore_rnn.set(sigmoidDetailViewInfo);

  let logitLayer = svg.select('.logit-layer');
  let logitLayerLower = svg.select('.underneath');
  let intermediateLayer = svg.select('.intermediate-layer');

  // Highlight the circle
  logitLayer.select(`#logit-circle-${i}`)
    .style('stroke-width', 2);

  // Highlight the associated plus symbol
  intermediateLayer.select(`#plus-symbol-clone-${i}`)
    .style('opacity', 1)
    .select('circle')
    .style('fill', d => d.fill);
  
  // Raise the associated edge group
  logitLayerLower.select(`#logit-lower-${i}`).raise();

  // Highlight the associated edges
  logitLayerLower.selectAll(`.sigmoid-abstract-edge-${i}`)
    .style('stroke-width', 0.8)
    .style('stroke', '#E0E0E0');

  logitLayerLower.selectAll(`.sigmoid-edge-${i}`)
    .style('stroke-width', 1)
    .style('stroke', '#E0E0E0');
  
  logitLayerLower.selectAll(`.logit-output-edge-${i}`)
    .style('stroke-width', 3)
    .style('stroke', '#E0E0E0');

  logitLayer.selectAll(`.logit-output-edge-${i}`)
    .style('stroke-width', 3)
    .style('stroke', '#E0E0E0');
}

const logitCircleMouseLeaveHandler = (i) => {
  // screenshot
  // return;

  // Update the hover info UI
  hoverInfoStore_rnn.set({
    show: false,
    text: `Logit: ${formater(logits[i])}`
  })

  // Dehighlight the text in the detail view
  sigmoidDetailViewInfo.highlightI = -1;
  sigmoidDetailViewStore_rnn.set(sigmoidDetailViewInfo);

  let logitLayer = svg.select('.logit-layer');
  let logitLayerLower = svg.select('.underneath');
  let intermediateLayer = svg.select('.intermediate-layer');

  // Restore the circle
  logitLayer.select(`#logit-circle-${i}`)
    .style('stroke-width', 1);

  // Restore the associated plus symbol
  intermediateLayer.select(`#plus-symbol-clone-${i}`)
    .style('opacity', 0.2);

  // Restore the associated edges
  logitLayerLower.selectAll(`.sigmoid-abstract-edge-${i}`)
    .style('stroke-width', 0.2)
    .style('stroke', '#EDEDED');

  logitLayerLower.selectAll(`.sigmoid-edge-${i}`)
    .style('stroke-width', 0.2)
    .style('stroke', '#F1F1F1');

  logitLayerLower.selectAll(`.logit-output-edge-${i}`)
    .style('stroke-width', 1.2)
    .style('stroke', '#E5E5E5');
  
  logitLayer.selectAll(`.logit-output-edge-${i}`)
    .style('stroke-width', 1.2)
    .style('stroke', '#E5E5E5');
}

// This function is binded to the detail view in Overview.svelte
export const sigmoidDetailViewMouseOverHandler = (event) => {
  logitCircleMouseOverHandler(event.detail.curI);
}

// This function is binded to the detail view in Overview.svelte
export const sigmoidDetailViewMouseLeaveHandler = (event) => {
  logitCircleMouseLeaveHandler(event.detail.curI);
}

const drawLogitLayer = (arg) => {
  let curLayerIndex = arg.curLayerIndex,
    moveX = arg.moveX,
    sigmoidLeftMid = arg.sigmoidLeftMid,
    selectedI = arg.selectedI,
    intermediateX1 = arg.intermediateX1,
    intermediateX2 = arg.intermediateX2,
    pixelWidth = arg.pixelWidth,
    pixelHeight = arg.pixelHeight,
    topY = arg.topY,
    bottomY = arg.bottomY,
    sigmoidX = arg.sigmoidX,
    middleGap = arg.middleGap,
    middleRectHeight = arg.middleRectHeight,
    symbolGroup = arg.symbolGroup,
    symbolX = arg.symbolX,
    lstmRange = arg.lstmRange;

  let logitLayer = svg.select('.intermediate-layer')
    .append('g')
    .attr('class', 'logit-layer')
    .raise();
  
  // Minotr layer ordering change
  let tempClone = svg.select('.intermediate-layer')
    .select('.dense-layer')
    .select('.plus-symbol')
    .clone(true)
    .attr('class', 'temp-clone-plus-symbol')
    .attr('transform', `translate(${symbolX - moveX},
      ${nodeCoordinate[curLayerIndex][selectedI].y + nodeHeight / 2})`)
    // Cool hack -> d3 clone doesnt clone events, make the front object pointer
    // event transparent so users can trigger the underlying object's event!
    .style('pointer-events', 'none')
    .remove();

  let tempPlusSymbol = logitLayer.append(() => tempClone.node());
  
  svg.select('.sigmoid-symbol').raise();

  let logitLayerLower = svg.select('.underneath')
    .append('g')
    .attr('class', 'logit-layer-lower')
    .lower();
  
  // Use circles to encode logit values
  let centerX = sigmoidLeftMid - moveX * 4 / 5;

  // Get all logits
  logits = [];
  let outputs = rnn[layerIndexDict['dense_Dense1']];
  for (let i = 0; i < outputs.length; i++) {
    logits.push(outputs[i].logit);
  }

  // Construct a color scale for the logit values
  let logitColorScale = d3.scaleLinear()
    .domain(d3.extent(logits))
    .range([0.2, 1]);
  
  // Draw the current logit circle before animation
  let logitRadius = 8;
  logitLayer.append('circle')
    .attr('class', 'logit-circle')
    .attr('id', `logit-circle-${selectedI}`)
    .attr('cx', centerX)
    .attr('cy', nodeCoordinate[curLayerIndex][selectedI].y + nodeHeight / 2)
    .attr('r', logitRadius)
    .style('fill', layerColorScales.logit(logitColorScale(logits[selectedI])))
    .style('cursor', 'crosshair')
    .style('pointer-events', 'all')
    .style('stroke', intermediateColor)
    .on('mouseover', () => logitCircleMouseOverHandler(selectedI))
    .on('mouseleave', () => logitCircleMouseLeaveHandler(selectedI))
    .on('click', () => { d3.event.stopPropagation() });
  
  // Show the logit circle corresponding label
  let sigmoidDetailAnnotation = svg.select('.intermediate-layer-annotation')
    .select('.sigmoid-detail-annoataion');

  sigmoidDetailAnnotation.select(`#logit-text-${selectedI}`)
    .style('opacity', 1);

  tempPlusSymbol.raise();

  // Draw another line from plus symbol to sigmoid symbol
  logitLayer.append('line')
    .attr('class', `logit-output-edge-${selectedI}`)
    .attr('x1', intermediateX2 - moveX + plusSymbolRadius * 2)
    .attr('x2', sigmoidX)
    .attr('y1', nodeCoordinate[curLayerIndex][selectedI].y + nodeHeight / 2)
    .attr('y2', nodeCoordinate[curLayerIndex][selectedI].y + nodeHeight / 2)
    .style('fill', 'none')
    .style('stroke', '#EAEAEA')
    .style('stroke-width', '1.2')
    .lower();

  // Add the dense to logit links
  let linkData = [];
  let lstmLength = rnn.lstm.length;
  let underneathIs = [...Array(outputs.length).keys()]
    .filter(d => d != selectedI);
  console.log(underneathIs);
  let curIIndex = 0;
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  const drawOneEdgeGroup = () => {
    // Only draw the new group if it is in the sigmoid mode
    if (!allowsSigmoidAnimation) {
      svg.select('.underneath')
        .selectAll(`.logit-lower`)
        .remove();
      return;
    }

    if (underneathIs.length ===0) {
      return;
    }

    let curI = underneathIs[curIIndex];

    let curEdgeGroup = svg.select('.underneath')
      .select(`#logit-lower-${curI}`);
    
    if (curEdgeGroup.empty()) {
      curEdgeGroup = svg.select('.underneath')
        .append('g')
        .attr('class', 'logit-lower')
        .attr('id', `logit-lower-${curI}`)
        .style('opacity', 0);

      // Hack: now show all edges, only draw 1/3 of the actual edges
      for (let f = 0; f < lstmLength; f += 3) {
        // let loopFactors = [0, 9];
        // loopFactors.forEach(l => {
        //   let factoredF = f + l * lstmLength;
    
          // dense -> output
          linkData.push({
            source: {x: intermediateX1 + pixelWidth + 3 - moveX,
              y:  nodeCoordinate[curLayerIndex-1][f]+ nodeHeight / 2},
            target: {x: intermediateX2 - moveX,
              y: nodeCoordinate[curLayerIndex][curI].y + nodeHeight / 2},
            index: factoredF,
            weight: rnn.lstm[factoredF].outputLinks[curI].weight,
            color: '#F1F1F1',
            width: 0.5,
            opacity: 1,
            class: `sigmoid-edge-${curI}`
          });
        // });
      }

      // Draw middle rect to logits

      // for (let vi = 0; vi < outputs.length - 2; vi++) {
      //   linkData.push({
      //     source: {x: intermediateX1 + pixelWidth + 3 - moveX,
      //       y: topY + lstmLength * pixelHeight + middleGap * (vi + 1) +
      //       middleRectHeight * (vi + 0.5)},
      //     target: {x: intermediateX2 - moveX,
      //       y: nodeCoordinate[curLayerIndex][curI].y + nodeLength / 2},
      //     index: -1,
      //     color: '#EDEDED',
      //     width: 0.5,
      //     opacity: 1,
      //     class: `sigmoid-abstract-edge-${curI}`
      //   });
      // }

      // Render the edges on the underneath layer
      curEdgeGroup.selectAll(`path.sigmoid-edge-${curI}`)
        .data(linkData)
        .enter()
        .append('path')
        .attr('class', d => d.class)
        .attr('id', d => `edge-${d.name}`)
        .attr('d', d => linkGen({source: d.source, target: d.target}))
        .style('fill', 'none')
        .style('stroke-width', d => d.width)
        .style('stroke', d => d.color === undefined ? intermediateColor : d.color)
        .style('opacity', d => d.opacity)
        .style('pointer-events', 'none');
    }
    
    let curNodeGroup = logitLayer.append('g')
      .attr('class', `logit-layer-${curI}`)
      .style('opacity', 0);
    
    // Draw the plus symbol
    let symbolClone = symbolGroup.clone(true)
      .style('opacity', 0);

    // Change the style of the clone
    symbolClone.attr('class', 'plus-symbol-clone')
      .attr('id', `plus-symbol-clone-${curI}`)
      .select('circle')
      .datum({fill: gappedColorScale(layerColorScales.weight,
        lstmRange, outputs[curI].bias, 0.35)})
      .style('pointer-events', 'none')
      .style('fill', '#E5E5E5');

    symbolClone.attr('transform', `translate(${symbolX},
      ${nodeCoordinate[curLayerIndex][curI].y + nodeLength / 2})`);
    
    // Draw the outter link using only merged path
    let outputEdgeD1 = linkGen({
      source: {
        x: intermediateX2 - moveX + plusSymbolRadius * 2,
        y: nodeCoordinate[curLayerIndex][curI].y + nodeLength / 2
      },
      target: {
        x: centerX + logitRadius,
        y: nodeCoordinate[curLayerIndex][curI].y + nodeLength / 2
      }
    });

    let outputEdgeD2 = linkGen({
      source: {
        x: centerX + logitRadius,
        y: nodeCoordinate[curLayerIndex][curI].y + nodeLength / 2
      },
      target: {
        x: sigmoidX,
        y: nodeCoordinate[curLayerIndex][selectedI].y + nodeLength / 2
      }
    });

    // There are ways to combine these two paths into one. However, the animation
    // for merged path is not continuous, so we use two saperate paths here.

    let outputEdge1 = logitLayerLower.append('path')
      .attr('class', `logit-output-edge-${curI}`)
      .attr('d', outputEdgeD1)
      .style('fill', 'none')
      .style('stroke', '#EAEAEA')
      .style('stroke-width', '1.2');

    let outputEdge2 = logitLayerLower.append('path')
      .attr('class', `logit-output-edge-${curI}`)
      .attr('d', outputEdgeD2)
      .style('fill', 'none')
      .style('stroke', '#EAEAEA')
      .style('stroke-width', '1.2');
    
    let outputEdgeLength1 = outputEdge1.node().getTotalLength();
    let outputEdgeLength2 = outputEdge2.node().getTotalLength();
    let totalLength = outputEdgeLength1 + outputEdgeLength2;
    let totalDuration = hasInitialized ? 500 : 800;
    let opacityDuration = hasInitialized ? 400 : 600;

    outputEdge1.attr('stroke-dasharray', outputEdgeLength1 + ' ' + outputEdgeLength1)
      .attr('stroke-dashoffset', outputEdgeLength1);
    
    outputEdge2.attr('stroke-dasharray', outputEdgeLength2 + ' ' + outputEdgeLength2)
      .attr('stroke-dashoffset', outputEdgeLength2);

    outputEdge1.transition('sigmoid-output-edge')
      .duration(outputEdgeLength1 / totalLength * totalDuration)
      .attr('stroke-dashoffset', 0);

    outputEdge2.transition('sigmoid-output-edge')
      .delay(outputEdgeLength1 / totalLength * totalDuration)
      .duration(outputEdgeLength2 / totalLength * totalDuration)
      .attr('stroke-dashoffset', 0);
    
    // Draw the logit circle
    curNodeGroup.append('circle')
      .attr('class', 'logit-circle')
      .attr('id', `logit-circle-${curI}`)
      .attr('cx', centerX)
      .attr('cy', nodeCoordinate[curLayerIndex - 1][curI].y + nodeLength / 2)
      .attr('r', 7)
      .style('fill', layerColorScales.logit(logitColorScale(logits[curI])))
      .style('stroke', intermediateColor)
      .style('cursor', 'crosshair')
      .on('mouseover', () => logitCircleMouseOverHandler(curI))
      .on('mouseleave', () => logitCircleMouseLeaveHandler(curI))
      .on('click', () => { d3.event.stopPropagation() });
    
    // Show the element in the detailed view
    sigmoidDetailViewInfo.startAnimation = {
      i: curI,
      duration: opacityDuration,
      // Always show the animation
      hasInitialized: false
    };
    sigmoidDetailViewStore_rnn.set(sigmoidDetailViewInfo);

    // Show the elements with animation    
    curNodeGroup.transition('sigmoid-edge')
      .duration(opacityDuration)
      .style('opacity', 1);

    if ((selectedI < 3 && curI == 9) || (selectedI >= 3 && curI == 0)) {
      // Show the hover text
      sigmoidDetailAnnotation.select('.sigmoid-detail-hover-annotation')
        .transition('sigmoid-edge')
        .duration(opacityDuration)
        .style('opacity', 1);
    }

    sigmoidDetailAnnotation.select(`#logit-text-${curI}`)
      .transition('sigmoid-edge')
      .duration(opacityDuration)
      .style('opacity', 1);
    
    curEdgeGroup.transition('sigmoid-edge')
      .duration(opacityDuration)
      .style('opacity', 1)
      .on('end', () => {
        // Recursive animaiton
        curIIndex ++;
        if (curIIndex < underneathIs.length) {
          linkData = [];
          drawOneEdgeGroup();
        } else {
          hasInitialized = true;
          sigmoidDetailViewInfo.hasInitialized = true;
          sigmoidDetailViewStore_rnn.set(sigmoidDetailViewInfo);
        }
      });
    
    symbolClone.transition('sigmoid-edge')
      .duration(opacityDuration)
      .style('opacity', 0.2);
  }

  // Show the sigmoid detail view
  let anchorElement = svg.select('.intermediate-layer')
    .select('.layer-label').node();
  let pos = getMidCoords(svg, anchorElement);
  let wholeSvg = d3.select('#rnn-svg');
  let svgYMid = +wholeSvg.style('height').replace('px', '') / 2;
  let detailViewTop = 2.5*nodeLength + svgYMid;

  const detailview = document.getElementById('detailview');
  detailview.style.top = `${detailViewTop}px`;
  detailview.style.left = `${pos.left -  nodeLength}px`;
  detailview.style.position = 'absolute';

  sigmoidDetailViewStore_rnn.set({
    show: true,
    logits: logits,
    logitColors: logits.map(d => layerColorScales.logit(logitColorScale(d))),
    selectedI: selectedI,
    highlightI: -1,
    outputName: classList[selectedI],
    outputValue: outputs[selectedI].output,
    startAnimation: {i: -1, duration: 0, hasInitialized: hasInitialized}
  })

  drawOneEdgeGroup();

  // same as input nodes
  let legendHeight = inputNodeHeight;
  // vSpaceAroundGap for each layer varies because of various number of nodes
  legendY = svgPaddings.top + vSpaceAroundGap 
        * (rnn[rnn.length-1].length+1) + 3 * legendHeight;

  let legendRange = 
  // Draw logit circle color scale
  drawIntermediateLayerLegend({
    legendHeight: 5,
    curLayerIndex: curLayerIndex,
    range: Math.abs(d3.extent(logits)[1]) + Math.abs(d3.extent(logits)[0]),
    minMax: {min: -Math.abs(d3.extent(logits)[0]), max: Math.abs(d3.extent(logits)[1])},
    group: logitLayer,
    width: sigmoidX - (intermediateX2 + plusSymbolRadius * 2 - moveX + 5),
    gradientAppendingName: 'flatten-logit-gradient',
    gradientGap: 0.1,
    colorScale: layerColorScales.logit,
    x: intermediateX2 + plusSymbolRadius * 2 - moveX + 5,
    y: legendY
  });

  // Draw logit layer label
  let logitLabel = logitLayer.append('g')
    .attr('class', 'layer-label')
    .classed('hidden', detailedMode)
    .attr('transform', () => {
      let x = centerX;
      let y = labelY;
      return `translate(${x}, ${y})`;
    });

  logitLabel.append('text')
    .style('text-anchor', 'middle')
    .style('dominant-baseline', 'middle')
    .style('opacity', 0.8)
    .style('font-weight', 800)
    .text('logit');
}

const removeLogitLayer = () => {
  svg.select('.logit-layer').remove();
  svg.select('.logit-layer-lower').remove();
  svg.selectAll('.plus-symbol-clone').remove();

  // Instead of removing the paths, we hide them, so it is faster to load in
  // the future
  svg.select('.underneath')
    .selectAll('.logit-lower')
    .style('opacity', 0);

  sigmoidDetailViewStore_rnn.set({
      show: false,
      logits: []
  })
}

const sigmoidClicked = (arg) => {
  let curLayerIndex = arg.curLayerIndex,
    moveX = arg.moveX,
    symbolX = arg.symbolX,
    symbolY = arg.symbolY,
    outputX = arg.outputX,
    outputY = arg.outputY,
    sigmoidLeftMid = arg.sigmoidLeftMid,
    selectedI = arg.selectedI,
    intermediateX1 = arg.intermediateX1,
    intermediateX2 = arg.intermediateX2,
    pixelWidth = arg.pixelWidth,
    pixelHeight = arg.pixelHeight,
    topY = arg.topY,
    bottomY = arg.bottomY,
    middleGap = arg.middleGap,
    middleRectHeight = arg.middleRectHeight,
    sigmoidX = arg.sigmoidX,
    sigmoidTextY = arg.sigmoidTextY,
    sigmoidWidth = arg.sigmoidWidth,
    symbolGroup = arg.symbolGroup,
    lstmRange = arg.lstmRange;

  let duration = 600;
  let centerX = sigmoidLeftMid - moveX * 4 / 5;
  d3.event.stopPropagation();

  // Clean up the logit elemends before moving anything
  if (isInSigmoid) {
    allowsSigmoidAnimationStore_rnn.set(false);
    removeLogitLayer();
  } else {
    allowsSigmoidAnimationStore_rnn.set(true);
  }

  // Move the overlay gradient
  svg.select('.intermediate-layer-overlay')
    .select('rect.overlay')
    .transition('sigmoid')
    .ease(d3.easeCubicInOut)
    .duration(duration)
    .attr('transform', `translate(${isInSigmoid ? 0 : -moveX}, ${0})`);

  // Move the legends
  svg.selectAll(`.intermediate-legend-${curLayerIndex - 1}`)
    .each((d, i, g) => moveLegend(d, i, g, moveX, duration, isInSigmoid));

  svg.select('.intermediate-layer')
    .select(`.layer-label`)
    .each((d, i, g) => moveLegend(d, i, g, moveX, duration, isInSigmoid));

  svg.select('.intermediate-layer')
    .select(`.layer-detailed-label`)
    .each((d, i, g) => moveLegend(d, i, g, moveX, duration, isInSigmoid));

  // Also move all layers on the left
  for (let i = curLayerIndex - 1; i >= 0; i--) {
    let curLayer = svg.select(`g#rnn-layer-group-${i}`);
    let previousX = +curLayer.select('image').attr('x');
    let newX = isInSigmoid ? previousX + moveX : previousX - moveX;
    moveLayerX({
      layerIndex: i,
      targetX: newX,
      disable: true,
      delay: 0,
      transitionName: 'sigmoid',
      duration: duration
    });
  }

  // Hide the sum up annotation
  svg.select('.plus-annotation')
    .transition('sigmoid')
    .duration(duration)
    .style('opacity', isInSigmoid ? 1 : 0)
    .style('pointer-events', isInSigmoid ? 'all' : 'none');

  // Hide the sigmoid annotation
  let sigmoidAnnotation = svg.select('.sigmoid-annotation')
    .style('pointer-events', isInSigmoid ? 'all' : 'none');
  
  let sigmoidDetailAnnotation = sigmoidAnnotation.selectAll('.sigmoid-detail-annoataion')
    .data([0])
    .enter()
    .append('g')
    .attr('class', 'sigmoid-detail-annoataion');

  // Remove the detailed annoatioan when quitting the detail view
  if (isInSigmoid) {
    sigmoidAnnotation.selectAll('.sigmoid-detail-annoataion').remove();
  }

  sigmoidAnnotation.select('.arrow-group')
    .transition('sigmoid')
    .duration(duration)
    .style('opacity', isInSigmoid ? 1 : 0);

  sigmoidAnnotation.select('.annotation-text')
    .style('cursor', 'help')
    .style('pointer-events', 'all')
    .on('click', () => {
      d3.event.stopPropagation();
      // Scroll to the article element
      document.querySelector(`#article-sigmoid`).scrollIntoView({ 
        behavior: 'smooth' 
      });
    })
    .transition('sigmoid')
    .duration(duration)
    .style('opacity', isInSigmoid ? 1 : 0)
    .on('end', () => {
      if (!isInSigmoid) {
        // Add new annotation for the sigmoid button
        let textX = sigmoidX + sigmoidWidth / 2;
        let textY = sigmoidTextY - 10;

        if (selectedI === 0) {
          textY = sigmoidTextY + 70;
        }

        let text = sigmoidDetailAnnotation.append('text')
          .attr('x', textX)
          .attr('y', textY)
          .attr('class', 'annotation-text sigmoid-detail-text')
          .style('dominant-baseline', 'baseline')
          .style('text-anchor', 'middle')
          .text('Normalize ');
        
        text.append('tspan') 
          .attr('dx', 1)
          .style('fill', '#E56014')
          .text('logits');
        
        text.append('tspan')
          .attr('dx', 1)
          .text(' into');

        text.append('tspan')
          .attr('x', textX)
          .attr('dy', '1.1em')
          .text('class probabilities');

        if (selectedI === 0) {
          drawArrow({
            group: sigmoidDetailAnnotation,
            sx: sigmoidX + sigmoidWidth / 2 - 5,
            sy: sigmoidTextY + 44,
            tx: sigmoidX + sigmoidWidth / 2,
            ty: textY - 12,
            dr: 50,
            hFlip: true,
            marker: 'marker-alt'
          });
        } else {
          drawArrow({
            group: sigmoidDetailAnnotation,
            sx: sigmoidX + sigmoidWidth / 2 - 5,
            sy: sigmoidTextY + 4,
            tx: sigmoidX + sigmoidWidth / 2,
            ty: symbolY - plusSymbolRadius - 4,
            dr: 50,
            hFlip: true,
            marker: 'marker-alt'
          });
        }

        // Add annotation for the logit layer label
        textX = centerX + 45;
        textY = labelY  + 5;
        let arrowTX = centerX + 20;
        let arrowTY = labelY  + 5;

        sigmoidDetailAnnotation.append('g')
          .attr('class', 'layer-detailed-label')
          .attr('transform', () => {
            let x = centerX;
            let y = labelY - 5;
            return `translate(${x}, ${y})`;
          })
          .classed('hidden', !detailedMode)
          .append('text')
          .attr('y', labelY)
          .style('opacity', 0.7)
          .style('dominant-baseline', 'middle')
          .style('font-size', '12px')
          .style('font-weight', '800')
          .append('tspan')
          .attr('x', 0)
          .text('logit')
          .append('tspan')
          .attr('x', 0)
          .style('font-size', '8px')
          .style('font-weight', 'normal')
          .attr('dy', '1.5em')
          .text(`(${1})`);

        sigmoidDetailAnnotation.append('text')
          .attr('class', 'annotation-text')
          .attr('x', textX)
          .attr('y', labelY  + 5)
          .style('text-anchor', 'start')
          .text('Before')
          .append('tspan')
          .attr('x', textX)
          .attr('dy', '1em')
          .text('normalization')


        drawArrow({
          group: sigmoidDetailAnnotation,
          tx: arrowTX,
          ty: arrowTY,
          sx: textX - 6,
          sy: textY + 2,
          dr: 60,
          hFlip: false,
          marker: 'marker-alt'
        });

        let outputIndex = layerIndexDict['dense_Dense1'];

        sigmoidDetailAnnotation.append('text')
          .attr('class', 'annotation-text')
          .attr('x', nodeCoordinate[outputIndex][0].x - 30)
          .attr('y', labelY  + 3)
          .style('text-anchor', 'end')
          .text('After')
          .append('tspan')
          .attr('x', nodeCoordinate[outputIndex][0].x - 30)
          .attr('dy', '1em')
          .text('normalization')

        drawArrow({
          group: sigmoidDetailAnnotation,
          tx: nodeCoordinate[outputIndex][0].x ,
          ty: arrowTY,
          sx: nodeCoordinate[outputIndex][0].x - 24,
          sy: textY + 2,
          dr: 60,
          hFlip: true,
          marker: 'marker-alt'
        });

        // // Add annotation for the logit circle
        // for (let i = 0; i < 10; i++) {
        //   sigmoidDetailAnnotation.append('text')
        //     .attr('x', centerX)
        //     .attr('y', nodeCoordinate[curLayerIndex - 1][i].y + nodeLength / 2 + 8)
        //     .attr('class', 'annotation-text sigmoid-detail-text')
        //     .attr('id', `logit-text-${i}`)
        //     .style('text-anchor', 'middle')
        //     .style('dominant-baseline', 'hanging')
        //     .style('opacity', 0)
        //     .text(`${classList[i]}`);
        // }

        let hoverTextGroup = sigmoidDetailAnnotation.append('g')
          .attr('class', 'sigmoid-detail-hover-annotation')
          .style('opacity', 0);

        textX = centerX + 50;
        textY = nodeCoordinate[curLayerIndex - 1][0].y + nodeLength / 2;

        if (selectedI < 3) {
          textY = nodeCoordinate[curLayerIndex - 1][9].y + nodeLength / 2;
        }

        // Add annotation to prompt user to check the logit value
        let hoverText = hoverTextGroup.append('text')
          .attr('x', textX)
          .attr('y', textY)
          .attr('class', 'annotation-text sigmoid-detail-text sigmoid-hover-text')
          .style('text-anchor', 'start')
          .style('dominant-baseline', 'baseline')
          .append('tspan')
          .style('font-weight', 700)
          .style('dominant-baseline', 'baseline')
          .text(`Hover over `)
          .append('tspan')
          .style('font-weight', 400)
          .style('dominant-baseline', 'baseline')
          .text('to see');
        
        hoverText.append('tspan')
          .style('dominant-baseline', 'baseline')
          .attr('x', textX)
          .attr('dy', '1em')
          .text('its ');

        hoverText.append('tspan')
          .style('dominant-baseline', 'baseline')
          .attr('dx', 1)
          .style('fill', '#E56014')
          .text('logit');
        
        hoverText.append('tspan')
          .style('dominant-baseline', 'baseline')
          .attr('dx', 1)
          .text(' value');
        
        drawArrow({
          group: hoverTextGroup,
          tx: centerX + 15,
          ty: textY,
          sx: textX - 8,
          sy: textY + 2,
          dr: 60,
          hFlip: false
        });
      }
    })

  // Hide the annotation
  svg.select('.lstm-annotation')
    .transition('sigmoid')
    .duration(duration)
    .style('opacity', isInSigmoid ? 1 : 0)
    .style('pointer-events', isInSigmoid ? 'all' : 'none');

  // Move the left part of faltten layer elements
  let lstmLeftPart = svg.select('.lstm-layer-left');
  lstmLeftPart.transition('sigmoid')
    .duration(duration)
    .ease(d3.easeCubicInOut)
    .attr('transform', `translate(${isInSigmoid ? 0 : -moveX}, ${0})`)
    .on('end', () => {
      // Add the logit layer
      if (!isInSigmoid) {
        let logitArg = {
          curLayerIndex: curLayerIndex,
          moveX: moveX,
          sigmoidLeftMid: sigmoidLeftMid,
          selectedI: selectedI,
          intermediateX1: intermediateX1,
          intermediateX2: intermediateX2,
          pixelWidth: pixelWidth,
          pixelHeight: pixelHeight,
          topY: topY,
          bottomY: bottomY,
          middleGap: middleGap,
          middleRectHeight: middleRectHeight,
          sigmoidX: sigmoidX,
          symbolGroup: symbolGroup,
          symbolX: symbolX,
          lstmRange: lstmRange
        };
        drawLogitLayer(logitArg);
      }

      // Redraw the line from the plus symbol to the output node
      if (!isInSigmoid) {
        let newLine = lstmLeftPart.select('.edge-group')
          .append('line')
          .attr('class', 'symbol-output-line')
          .attr('x1', symbolX)
          .attr('y1', symbolY)
          .attr('x2', outputX + moveX)
          .attr('y2', outputY)
          .style('stroke-width', 1.2)
          .style('stroke', '#E5E5E5')
          .style('opacity', 0);
        
        newLine.transition('sigmoid')
          .delay(duration / 3)
          .duration(duration * 2 / 3)
          .style('opacity', 1);
      } else {
        lstmLeftPart.select('.symbol-output-line').remove();
      }
      
      isInSigmoid = !isInSigmoid;
      isInSigmoidStore_rnn.set(isInSigmoid);
    })
}

/**
 * Draw the lstm layer before output layer
 * @param {number} curLayerIndex Index of the selected layer
 * @param {object} d Bounded d3 data
 * @param {number} i Index of the selected node
 * @param {number} width rnn group width
 * @param {number} height rnn group height
 */
export const drawDense = (curLayerIndex, d, i, width, height) => {
  // Show the output legend
  svg.selectAll('.output-legend')
    .classed('hidden', false);

  let pixelWidth = nodeLength / 2;
  let pixelHeight = 1.1;
  console.log('hSpaceAroundGap is: ',hSpaceAroundGap)
  let totalLength = (2 * nodeLength +
    2 * hSpaceAroundGap * gapRatio + pixelWidth);
  console.log('total Length is: ', totalLength);
  let leftX = nodeCoordinate[curLayerIndex][0].x - totalLength;
  console.log('leftX is: ', leftX);
  let intermediateGap = (hSpaceAroundGap * gapRatio * 1) / 2;
  console.log('intermediateGap: ', intermediateGap);
  const minimumGap = 20;
  let linkGen = d3.linkHorizontal()
    .x(d => d.x)
    .y(d => d.y);

  // Hide the edges
  svg.select('g.edge-group')
    .style('visibility', 'hidden');

  // Move the previous layer
  moveLayerX({layerIndex: curLayerIndex - 1, targetX: leftX,
    disable: true, delay: 0});

  // Disable the current layer (output layer)
  moveLayerX({layerIndex: curLayerIndex,
    targetX: nodeCoordinate[curLayerIndex][0].x, disable: true,
    delay: 0, opacity: 0.15, specialIndex: i});
  
  // Compute the gap in the left shrink region
  let leftEnd = leftX - hSpaceAroundGap;
  let leftGap = (leftEnd - nodeCoordinate[0][0].x 
        - (numLayer-3) * nodeLength - embeedingLen) / (numLayer-2);
  console.log('leftEnd is: ', leftEnd);
  console.log('leftGap is: ',leftGap);

  // Different from other intermediate view, we push the left part dynamically
  // 1. If there is enough space, we fix the first layer position and move all
  // other layers;
  // 2. If there is not enough space, we maintain the minimum gap and push all
  // left layers to the left (could be out-of-screen)
  if (leftGap > minimumGap) {
    // Move the left layers
    for (let i = 0; i < curLayerIndex - 1; i++) {
      // let curX = nodeCoordinate[0][0].x + i * (nodeLength + leftGap);
      let curX;
      if (i<1) {
        curX = nodeCoordinate[0][0].x + i * (nodeLength + leftGap);
      } else {
        curX = nodeCoordinate[0][0].x + i * leftGap + (i-1)*nodeLength + embeedingLen
      }
      moveLayerX({layerIndex: i, targetX: curX, disable: true, delay: 0});
    }
  } else {
    leftGap = minimumGap;
    let curLeftBound = leftX - leftGap * 2 - embeedingLen;
    // Move the left layers
    for (let i = curLayerIndex - 2; i >= 0; i--) {
      console.log('curLeftBound is: ',curLeftBound);
      moveLayerX({layerIndex: i, targetX: curLeftBound, disable: true, delay: 0});
      curLeftBound = curLeftBound - leftGap - nodeLength;
    }
  }

  // Add an overlay
  let stops = [{offset: '0%', color: 'rgb(250, 250, 250)', opacity: 1},
    {offset: '50%', color: 'rgb(250, 250, 250)', opacity: 0.95},
    {offset: '100%', color: 'rgb(250, 250, 250)', opacity: 0.85}];
  addOverlayGradient('overlay-gradient-left', stops);

  let intermediateLayerOverlay = svg.append('g')
    .attr('class', 'intermediate-layer-overlay');

  intermediateLayerOverlay.append('rect')
    .attr('class', 'overlay')
    .style('fill', 'url(#overlay-gradient-left)')
    .style('stroke', 'none')
    .attr('width', leftX + svgPaddings.left - (leftGap * 2) + 3)
    .attr('height', height + svgPaddings.top + svgPaddings.bottom)
    .attr('x', -svgPaddings.left)
    .attr('y', 0)
    .style('opacity', 0);
  
  intermediateLayerOverlay.selectAll('rect.overlay')
    .transition('move')
    .duration(800)
    .ease(d3.easeCubicInOut)
    .style('opacity', 1);

  // Add the intermediate layer
  let intermediateLayer = svg.append('g')
    .attr('class', 'intermediate-layer')
    .style('opacity', 0);
  
  let intermediateX1 = leftX + nodeLength + intermediateGap;
  console.log('intermediateX1 is: ', intermediateX1)
  let intermediateX2 = intermediateX1 + intermediateGap + pixelWidth;
  console.log('intermediateX2 is: ',intermediateX2);
  let preLayerRange = rnnLayerRanges[selectedScaleLevel][curLayerIndex - 1];
  let colorScale = layerColorScales.conv;
  // let lstmLength = rnn.lstm.length / rnn[1].length;
    let lstmLength = rnn.lstm.length;
  let linkData = [];

  let denseLayer = intermediateLayer.append('g')
    .attr('class', 'dense-layer');
  
  let lstmLayerLeftPart = denseLayer.append('g')
    .attr('class', 'lstm-layer-left');
  
  let topY = nodeCoordinate[curLayerIndex - 1][0].y;
  let lastNodeIndexOfPreLayer = rnn[curLayerIndex-1].length-1
  let bottomY = nodeCoordinate[curLayerIndex - 1][lastNodeIndexOfPreLayer].y + nodeHeight -
        lstmLength * pixelHeight;
  
  // Compute the pre-layer gap
  // let preLayerDimension = rnn[curLayerIndex - 1][0].output.length;
  // lstm output is number, so the length is 1
  let preLayerDimension = 1;
  let preLayerGap = nodeLength / (2 * preLayerDimension);

  // Compute bounding box length
  let boundingBoxLength = nodeHeight / preLayerDimension;

  let lstmRange = rnnLayerRanges[selectedScaleLevel][curLayerIndex-1];

  let lstmMouseOverHandler = (d) => {
    let index = d.index;
    // Screenshot
    // console.log(index);

    // Update the hover info UI
    if (d.weight === undefined) {
      hoverInfo = {
        show: true,
        text: `lstm output: ${formater(rnn.lstm[index].output)}`
      };
    } else {
      hoverInfo = {
        show: true,
        text: `Weight: ${formater(d.weight)}`
      };
    }
    hoverInfoStore_rnn.set(hoverInfo);

    // lstmLayerLeftPart.select(`#edge-lstm-${index}`)
    //   .raise()
    //   .style('stroke', intermediateColor)
    //   .style('stroke-width', 1);

    lstmLayerLeftPart.select(`#edge-lstm-${index}-output`)
      .raise()
      .style('stroke-width', 1)
      .style('stroke', da => gappedColorScale(layerColorScales.weight,
        lstmRange, da.weight, 0.1));

    lstmLayerLeftPart.select(`#bounding-${index}`)
      .raise()
      .style('opacity', 1);
  }

  let lstmMouseLeaveHandler = (d) => {
    let index = d.index;

    // screenshot
    // if (index === 70) {return;}

    // Update the hover info UI
    if (d.weight === undefined) {
      hoverInfo = {
        show: false,
        text: `lstm output: ${formater(rnn.lstm[index].output)}`
      };
    } else {
      hoverInfo = {
        show: false,
        text: `Weight: ${formater(d.weight)}`
      };
    }
    hoverInfoStore_rnn.set(hoverInfo);

    // lstmLayerLeftPart.select(`#edge-lstm-${index}`)
    //   .style('stroke-width', 0.6)
    //   .style('stroke', '#E5E5E5')

    lstmLayerLeftPart.select(`#edge-lstm-${index}-output`)
      .style('stroke-width', 0.6)
      .style('stroke', da => gappedColorScale(layerColorScales.weight,
        lstmRange, da.weight, 0.35));

    lstmLayerLeftPart.select(`#bounding-${index}`)
      .raise()
      .style('opacity', 0);
  }

  for (let f = 0; f < lstmLength; f++) {
      // lstm -> output
      linkData.push({
        source: {x: leftX + nodeLength + 3,
          y: nodeCoordinate[curLayerIndex-1][f].y + nodeHeight / 2},
        target: {x: intermediateX2,
          //nodeCoordinate[curLayerIndex][i].x - nodeLength,
          y: nodeCoordinate[curLayerIndex][i].y + nodeHeight / 2},
        index: f,
        weight: rnn.lstm[f].outputLinks[i].weight,
        name: `lstm-${f}-output`,
        color: gappedColorScale(layerColorScales.weight,
          lstmRange, rnn.lstm[f].outputLinks[i].weight, 0.35),
        width: 0.6,
        opacity: 1,
        class: `lstm-output`
      });

      // Add original pixel bounding box
      let loc = rnn.lstm[f].outputLinks[0].weight;
      lstmLayerLeftPart.append('rect')
        .attr('id', `bounding-${f}`)
        .attr('class', 'lstm-bounding')
        // .attr('x', leftX + loc[1] * boundingBoxLength)
        .attr('x', leftX)
        // .attr('y', nodeCoordinate[curLayerIndex - 1][l].y + loc[0] * boundingBoxLength)
        .attr('y', nodeCoordinate[curLayerIndex - 1][f].y )
        // .attr('width', boundingBoxLength)
        .attr('width', nodeLength)
        .attr('height', boundingBoxLength)
        .style('fill', 'none')
        .style('stroke', intermediateColor)
        .style('stroke-length', '0.2')
        .style('pointer-events', 'all')
        .style('cursor', 'crosshair')
        .style('opacity', 0)
        .on('mouseover', () => lstmMouseOverHandler({index: f}))
        .on('mouseleave', () => lstmMouseLeaveHandler({index: f}))
        .on('click', () => {d3.event.stopPropagation()});
  }
  
  // Compute the middle gap
  let middleGap = 5;
  // let middleRectHeight = (10 * nodeLength + (10 - 1) * vSpaceAroundGap -
  //   pixelHeight * lstmLength * 2 - 5 * (8 + 1)) / 8;
  let middleRectHeight = (height)

  // Draw the plus operation symbol
  let symbolX = intermediateX2 + plusSymbolRadius;
  let symbolY = nodeCoordinate[curLayerIndex][i].y + nodeHeight / 2;
  let symbolRectHeight = 1;
  let symbolGroup = lstmLayerLeftPart.append('g')
    .attr('class', 'plus-symbol')
    .attr('transform', `translate(${symbolX}, ${symbolY})`);
  
  symbolGroup.append('rect')
    .attr('x', -plusSymbolRadius)
    .attr('y', -plusSymbolRadius)
    .attr('width', plusSymbolRadius * 2)
    .attr('height', plusSymbolRadius * 2)
    .attr('rx', 3)
    .attr('ry', 3)
    .style('fill', 'none')
    .style('stroke', intermediateColor);
  
  symbolGroup.append('rect')
    .attr('x', -(plusSymbolRadius - 3))
    .attr('y', -symbolRectHeight / 2)
    .attr('width', 2 * (plusSymbolRadius - 3))
    .attr('height', symbolRectHeight)
    .style('fill', intermediateColor);

  symbolGroup.append('rect')
    .attr('x', -symbolRectHeight / 2)
    .attr('y', -(plusSymbolRadius - 3))
    .attr('width', symbolRectHeight)
    .attr('height', 2 * (plusSymbolRadius - 3))
    .style('fill', intermediateColor);

  // Place the bias rectangle below the plus sign if user clicks the first
  // conv node (no need now, since we added annotaiton for sigmoid to make it
  // look better aligned)
  // Add bias symbol to the plus symbol
  symbolGroup.append('circle')
    .attr('cx', 0)
    .attr('cy', -nodeLength / 2 - 0.5 * kernelRectLength)
    .attr('r', kernelRectLength * 1.5)
    .style('stroke', intermediateColor)
    .style('cursor', 'crosshair')
    .style('fill', gappedColorScale(layerColorScales.weight,
        lstmRange, d.bias, 0.35))
    .on('mouseover', () => {
      hoverInfoStore_rnn.set( {show: true, text: `Bias: ${formater(d.bias)}`} );
    })
    .on('mouseleave', () => {
      hoverInfoStore_rnn.set( {show: false, text: `Bias: ${formater(d.bias)}`} );
    })
    .on('click', () => { d3.event.stopPropagation(); });
  
  // Link from bias to the plus symbol
  symbolGroup.append('path')
    .attr('d', linkGen({
      source: { x: 0, y: 0 },
      target: { x: 0, y: -nodeLength / 2 - 0.5 * kernelRectLength }
    }))
    .attr('id', 'bias-plus')
    .attr('stroke-width', 1.2)
    .attr('stroke', '#E5E5E5')
    .lower();

  // Link from the plus symbol to the output
  linkData.push({
    source: getOutputKnot({x: intermediateX2 + 2 * plusSymbolRadius - nodeLength,
      y: nodeCoordinate[curLayerIndex][i].y- nodeLength/2 +nodeHeight/2}),
    target: getInputKnot({x: nodeCoordinate[curLayerIndex][i].x - 3,
      y: nodeCoordinate[curLayerIndex][i].y - nodeLength/2 +nodeHeight/2}),
    name: `symbol-output`,
    width: 1.2,
    color: '#E5E5E5'
  });

  // Draw sigmoid operation symbol
  let sigmoidWidth = 55;
  let emptySpace = ((totalLength - 2 * nodeLength - 2 * intermediateGap)
    - sigmoidWidth) / 2;
  let symbolEndX = intermediateX2 + plusSymbolRadius * 2;
  let sigmoidX = emptySpace + symbolEndX;
  let sigmoidLeftMid = emptySpace / 2 + symbolEndX;
  let sigmoidTextY = nodeCoordinate[curLayerIndex][i].y - 2 * kernelRectLength - 2 * nodeHeight;
  let moveX = (intermediateX2 - (intermediateX1 + pixelWidth + 3)) * 2 / 3;

  let sigmoidArg = {
    curLayerIndex: curLayerIndex,
    moveX: moveX,
    symbolX: symbolX,
    symbolY: symbolY,
    outputX: nodeCoordinate[curLayerIndex][i].x,
    outputY: symbolY,
    sigmoidLeftMid: sigmoidLeftMid,
    selectedI: i,
    intermediateX1: intermediateX1,
    intermediateX2: intermediateX2,
    pixelWidth: pixelWidth,
    pixelHeight: pixelHeight,
    topY: topY,
    bottomY: bottomY,
    middleGap: middleGap,
    middleRectHeight: middleRectHeight,
    sigmoidX: sigmoidX,
    sigmoidWidth: sigmoidWidth,
    sigmoidTextY: sigmoidTextY,
    symbolGroup: symbolGroup,
    lstmRange: lstmRange
  };

  let sigmoidSymbol = intermediateLayer.append('g')
    .attr('class', 'sigmoid-symbol')
    .attr('transform', `translate(${sigmoidX}, ${symbolY})`)
    .style('pointer-event', 'all')
    .style('cursor', 'pointer')
    .on('click', () => sigmoidClicked(sigmoidArg));
  
  sigmoidSymbol.append('rect')
    .attr('x', 0)
    .attr('y', -plusSymbolRadius)
    .attr('width', sigmoidWidth)
    .attr('height', plusSymbolRadius * 2)
    .attr('stroke', intermediateColor)
    .attr('rx', 2)
    .attr('ry', 2)
    .attr('fill', '#FAFAFA');
  
  sigmoidSymbol.append('text')
    .attr('x', 5)
    .attr('y', 1)
    .style('dominant-baseline', 'middle')
    .style('font-size', '12px')
    .style('opacity', 0.5)
    .text('Sigmoid');

  // Draw the layer label
  let layerLabel = intermediateLayer.append('g')
    .attr('class', 'layer-label')
    .classed('hidden', detailedMode)
    .attr('transform', () => {
      // let x = leftX + nodeLength + (4 * hSpaceAroundGap * gapRatio +
      //   pixelWidth) / 2;
      let x = sigmoidX;
      // let y = (svgPaddings.top + vSpaceAroundGap) / 2 + 5;
      let y = labelY ;
      return `translate(${x}, ${y})`;
    })
    .style('cursor', 'help')
    .on('click', () => {
      d3.event.stopPropagation();
      // Scroll to the article element
      document.querySelector(`#article-lstm`).scrollIntoView({ 
        behavior: 'smooth' 
      });
    });
  
  // layerLabel.append('text')
  //   .style('dominant-baseline', 'middle')
  //   .style('opacity', 0.8)
  //   .style('font-weight', 800)
  //   .text('Activiation');

  let svgHeight = Number(d3.select('#rnn-svg').style('height').replace('px', '')) + 150;
  let scroll = new SmoothScroll('a[href*="#"]', {offset: -svgHeight});
    
  // let detailedLabelGroup = intermediateLayer.append('g')
  //   .attr('transform', () => {
  //     // let x = leftX + nodeLength + (4 * hSpaceAroundGap * gapRatio + pixelWidth) / 2;
  //     let x = sigmoidX;
  //     // let y = (svgPaddings.top + vSpaceAroundGap) / 2 - 5;
  //     let y = labelY ;
  //     return `translate(${x}, ${y})`;
  //   })
  //   .attr('class', 'layer-detailed-label')
  //   .classed('hidden', !detailedMode)
  //   .style('cursor', 'help')
  //   .on('click', () => {
  //     d3.event.stopPropagation();
  //     // Scroll to the article element
  //     let anchor = document.querySelector(`#article-lstm`);
  //     scroll.animateScroll(anchor);
  //   });
  
  // detailedLabelGroup.append('title')
  //   .text('Move to article section');

  // let detailedLabelText = detailedLabelGroup.append('text')
  //   .style('text-anchor', 'middle')
  //   .style('dominant-baseline', 'middle')
  //   .style('opacity', '0.7')
  //   .style('font-weight', 800)
  //   .append('tspan')
  //   .text('Activation');
  
  // // let dimension = rnn[layerIndexDict['max_pool_2']].length * 
  // //   rnn[layerIndexDict['max_pool_2']][0].output.length *
  // //   rnn[layerIndexDict['max_pool_2']][0].output[0].length;
  // let dimension = 1;

  // detailedLabelText.append('tspan')
  //   .attr('x', 0)
  //   .attr('dy', '1.5em')
  //   .style('font-size', '8px')
  //   .style('font-weight', 'normal')
  //   .text(`(${dimension})`);

  // Add edges between nodes
  let edgeGroup = lstmLayerLeftPart.append('g')
    .attr('class', 'edge-group')
    .lower();
  
  let dashoffset = 0;
  
  edgeGroup.selectAll('path')
    .data(linkData)
    .enter()
    .append('path')
    .attr('class', d => d.class)
    .classed('flow-edge', d=>d.name !== 'output-next')
    .attr('id', d => `edge-${d.name}`)
    .attr('d', d => linkGen({source: d.source, target: d.target}))
    .style('fill', 'none')
    .style('stroke-width', d => d.width)
    .style('stroke', d => d.color === undefined ? intermediateColor : d.color)
    .style('opacity', d => d.opacity);

  edgeGroup.selectAll('path.flow-edge')
    .attr('stroke-dasharray', '4 2')
    .attr('stroke-dashoffset', 0)
    .each((d, i, g) => animateEdge(d, i, g, dashoffset - 1000));
  
  edgeGroup.selectAll('path.lstm-abstract-output')
    .lower();

  edgeGroup.selectAll('path.lstm,path.lstm-output')
    .style('cursor', 'crosshair')
    .style('pointer-events', 'all')
    .on('mouseover', lstmMouseOverHandler)
    .on('mouseleave', lstmMouseLeaveHandler)
    .on('click', () => { d3.event.stopPropagation() });
  
  // vSpaceAroundGap for each layer varies because of various number of nodes
  let legentY = svgPaddings.top + vSpaceAroundGap 
    * (rnn[rnn.length-1].length+1) + 3 * inputNodeHeight;
  // // Add legend
  // drawIntermediateLayerLegend({
  //   legendHeight: 5,
  //   curLayerIndex: curLayerIndex,
  //   range: preLayerRange,
  //   minMax: rnnLayerMinMax[2],
  //   group: intermediateLayer,
  //   width: intermediateGap + nodeLength - 3,
  //   x: leftX,
  //   y: legentY
  // });

  drawIntermediateLayerLegend({
    legendHeight: 5,
    curLayerIndex: curLayerIndex,
    range: lstmRange,
    minMax: rnnLayerMinMax[curLayerIndex-1],
    group: intermediateLayer,
    width: intermediateX2-leftX,
    gradientAppendingName: 'lstm-weight-gradient',
    gradientGap: 0.1,
    colorScale: layerColorScales.weight,
    // x: leftX + intermediateGap + nodeLength + pixelWidth + 3,
    x: leftX,
    y: legentY
  });

  // Add annotation to the intermediate layer
  let intermediateLayerAnnotation = svg.append('g')
    .attr('class', 'intermediate-layer-annotation')
    .style('opacity', 0);

  // Add annotation for the sum operation
  let plusAnnotation = intermediateLayerAnnotation.append('g')
    .attr('class', 'plus-annotation');
  
  // let textX = nodeCoordinate[curLayerIndex][i].x - 50;
  let textX = intermediateX2;
  let textY = nodeCoordinate[curLayerIndex][i].y + nodeLength +
    kernelRectLength * 3;
  let arrowSY = nodeCoordinate[curLayerIndex][i].y + nodeLength +
    kernelRectLength * 2;
  let arrowTY = nodeCoordinate[curLayerIndex][i].y + nodeHeight / 2 +
    plusSymbolRadius;

  if (i == 9) {
    textY -= 110;
    arrowSY -= 70;
    arrowTY -= 18;
  }

  let plusText = plusAnnotation.append('text')
    .attr('x', textX)
    .attr('y', textY)
    .attr('class', 'annotation-text')
    .style('dominant-baseline', 'hanging')
    .style('text-anchor', 'middle');
  
  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .text('Add up all products');
  
  plusText.append('tspan')
    .attr('x', textX)
    .attr('dy', '1em')
    .style('dominant-baseline', 'hanging')
    .text('(');

  plusText.append('tspan')
    .style('fill', '#66a3c8')
    .style('dominant-baseline', 'hanging')
    .text('element');

  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .text(' × ');

  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .style('fill', '#b58946')
    .text('weight');

  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .text(')');

  plusText.append('tspan')
    .attr('x', textX)
    .attr('dy', '1em')
    .style('dominant-baseline', 'hanging')
    .text('and then ');

  plusText.append('tspan')
    .style('dominant-baseline', 'hanging')
    .style('fill', '#479d94')
    .text('bias');
  
  drawArrow({
    group: plusAnnotation,
    sx: intermediateX2 - 2 * plusSymbolRadius - 3,
    sy: arrowSY,
    tx: intermediateX2 - 5,
    ty: arrowTY,
    dr: 30,
    hFlip: i === 9,
    marker: 'marker-alt'
  });

  // Add annotation for the bias
  let biasTextY = symbolY -nodeLength / 2 - kernelRectLength;
  biasTextY -= 2 * kernelRectLength + 4;
  
  lstmLayerLeftPart.append('text')
    .attr('class', 'annotation-text')
    .attr('x', intermediateX2 + plusSymbolRadius)
    .attr('y', biasTextY)
    .style('text-anchor', 'middle')
    .style('dominant-baseline', 'baseline')
    .text('Bias');
  
  // Add annotation for the sigmoid symbol
  let sigmoidAnnotation = intermediateLayerAnnotation.append('g')
    .attr('class', 'sigmoid-annotation');
  
  sigmoidAnnotation.append('text')
    .attr('x', sigmoidX + sigmoidWidth / 2)
    .attr('y', sigmoidTextY)
    .attr('class', 'annotation-text')
    .style('dominant-baseline', 'baseline')
    .style('text-anchor', 'middle')
    .style('font-weight', 700)
    .text('Click ')
    .append('tspan')
    .attr('dx', 1)
    .style('font-weight', 400)
    .text('to learn more');

  drawArrow({
    group: sigmoidAnnotation,
    sx: sigmoidX + sigmoidWidth / 2 - 5,
    sy: sigmoidTextY + 4,
    tx: sigmoidX + sigmoidWidth / 2,
    ty: symbolY - plusSymbolRadius - 4,
    dr: 50,
    hFlip: true
  });

  // Add annotation for the output neuron
  let outputAnnotation = intermediateLayerAnnotation.append('g')
    .attr('class', 'output-annotation');

  let outputIndex = layerIndexDict['dense_Dense1'];
  
  outputAnnotation.append('text')
    .attr('x', nodeCoordinate[outputIndex][i].x)
    .attr('y', nodeCoordinate[outputIndex][i].y-nodeHeight)
    .attr('class', 'annotation-text')
    .text(`(${d3.format('.4f')(rnn[outputIndex][i].output)})`);


  /* Prototype of using arc to represent the dense layer (future)
  let pie = d3.pie()
    .padAngle(0)
    .sort(null)
    .value(d => d.output)
    .startAngle(0)
    .endAngle(-Math.PI);

  let radius = 490 / 2;
  let arc = d3.arc()
    .innerRadius(radius - 20)
    .outerRadius(radius);

  let arcs = pie(rnn.lstm);
  console.log(arcs);

  let test = svg.append('g')
    .attr('class', 'test')
    .attr('transform', 'translate(500, 250)');

  test.selectAll("path")
    .data(arcs)
    .join("path")
      .attr('class', 'arc')
      .attr("fill", d => colorScale((d.value + range/2) / range))
      .attr("d", arc);
  */

  // Show everything
  svg.selectAll('g.intermediate-layer, g.intermediate-layer-annotation')
    .transition()
    .delay(500)
    .duration(500)
    .ease(d3.easeCubicInOut)
    .style('opacity', 1);
}