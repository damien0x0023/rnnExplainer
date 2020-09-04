<script>
  // Svelte functions
  import { onMount } from 'svelte';
  import { 
    rnnStore, rnnLayerMinMaxStore, rnnLayerRangesStore, 
    svgStore_rnn, vSpaceAroundGapStore_rnn, hSpaceAroundGapStore_rnn,
    nodeCoordinateStore_rnn, selectedScaleLevelStore_rnn, needRedrawStore_rnn,
    detailedModeStore_rnn, shouldIntermediateAnimateStore_rnn, isInSigmoidStore_rnn, 
    sigmoidDetailViewStore_rnn, hoverInfoStore_rnn, allowsSigmoidAnimationStore_rnn, 
    intermediateLayerPositionStore_rnn, reviewArrayStore
  } from '../stores.js';

  import { Jumper } from 'svelte-loading-spinners';

  // Svelte views
  import SigmoidView from '../detail-view/Sigmoidview.svelte';
  import EmbeddingView from '../detail-view/Embeddingview.svelte';

  const LOCAL_URLS = {
    model: 'PUBLIC_URL/resources/model.json',
    metadata: 'PUBLIC_URL/resources/metadata.json'
  };

  // Overview functions
  import { loadTrainedModel_rnn, SentimentPredictor } from '../utils/rnn-tf.js';

  import { rnnOverviewConfig } from '../config.js';

  import {
    addOverlayRect, drawLstm
  } from './intermediate-drawRNN.js';

  import { moveLayerX, addOverlayGradient } from './intermediateRNN-utils.js';

  import {
    drawDense, sigmoidDetailViewMouseOverHandler, sigmoidDetailViewMouseLeaveHandler
  } from './Dense-draw.js';

  import {
    drawOutputRNN, drawRNN, updateRNN, updateRNNLayerRanges
  } from './overview-drawRNN.js';

  // View bindings
  let rnnOverviewComponent;
  let scaleLevelSet = new Set(['local', 'module', 'global']);
  let selectedScaleLevel = 'local';
  selectedScaleLevelStore_rnn.set(selectedScaleLevel);
  let previousSelectedScaleLevel = selectedScaleLevel;
  let overview = undefined;
  let wholeSvg_rnn = undefined;
  let svg = undefined;

  // let wholeSvg_cnn = undefined;
  // let svg_cnn = undefined;

  $: selectedScaleLevel, selectedScaleLevelChanged();

  // Configs
  const layerColorScales = rnnOverviewConfig.layerColorScales;
  const nodeLength = rnnOverviewConfig.nodeLength;
  const embbedingLength = rnnOverviewConfig.embedddingLength;
  const inputNodeHeight = rnnOverviewConfig.inputNodeHeight;
  const plusSymbolRadius = rnnOverviewConfig.plusSymbolRadius;
  const numLayers = rnnOverviewConfig.numLayers;
  const edgeOpacity = rnnOverviewConfig.edgeOpacity;
  const edgeInitColor = rnnOverviewConfig.edgeInitColor;
  const edgeHoverColor = rnnOverviewConfig.edgeHoverColor;
  const edgeHoverOuting = rnnOverviewConfig.edgeHoverOuting;
  const edgeStrokeWidth = rnnOverviewConfig.edgeStrokeWidth;
  const intermediateColor = rnnOverviewConfig.intermediateColor;
  const kernelRectLength = rnnOverviewConfig.kernelRectLength;
  const svgPaddings = rnnOverviewConfig.svgPaddings;
  const gapRatio = rnnOverviewConfig.gapRatio;
  const overlayRectOffset = rnnOverviewConfig.overlayRectOffset;
  const classLists = rnnOverviewConfig.classLists;

  // Shared properties
  // for rnn
  let needRedraw = [undefined, undefined];
  needRedrawStore_rnn.subscribe( value => {needRedraw = value;} );

  let nodeCoordinate = undefined;
  nodeCoordinateStore_rnn.subscribe( value => {nodeCoordinate = value;} )

  let rnnLayerRanges = undefined;
  rnnLayerRangesStore.subscribe( value => {rnnLayerRanges = value;} )

  let rnnLayerMinMax = undefined;
  rnnLayerMinMaxStore.subscribe( value => {rnnLayerMinMax = value;} )

  let detailedMode = undefined;
  detailedModeStore_rnn.subscribe( value => {detailedMode = value;} )

  let shouldIntermediateAnimate_rnn = undefined;
  shouldIntermediateAnimateStore_rnn.subscribe(value => {
    shouldIntermediateAnimate_rnn = value;
  })

  let vSpaceAroundGap = undefined;
  vSpaceAroundGapStore_rnn.subscribe( value => {vSpaceAroundGap = value;} )

  let hSpaceAroundGap = undefined;
  hSpaceAroundGapStore_rnn.subscribe( value => {hSpaceAroundGap = value;} )

  let isInSigmoid = undefined;
  isInSigmoidStore_rnn.subscribe( value => {isInSigmoid = value;} )

  let sigmoidDetailViewInfo = undefined;
  sigmoidDetailViewStore_rnn.subscribe( value => {
    sigmoidDetailViewInfo = value;
  } )

  let hoverInfo_rnn = undefined;
  hoverInfoStore_rnn.subscribe( value => {hoverInfo_rnn = value;} )

  let intermediateLayerPosition_rnn = undefined;
  intermediateLayerPositionStore_rnn.subscribe ( value => {
    intermediateLayerPosition_rnn = value;} )

  let reviewArray = undefined;
  reviewArrayStore.subscribe( value=> {reviewArray = value;})

  let width = undefined;
  let height = undefined;
  // let model = undefined;
  // lstm
  let model_lstm = undefined;
  let selectedNode = {layerName: '', index: -1, data: null};
  let isInIntermediateView = false;
  let isInActPoolDetailView = false;
  let actPoolDetailViewNodeIndex = -1;
  let actPoolDetailViewLayerIndex = -1;
  let detailedViewNum = undefined;
  let disableControl = false;

  // Wait to load
  let rnn = undefined;
  // let isRNNloaded = false;

  let detailedViewAbsCoords = {
    1 : [600, 270, 490, 290],
    2: [500, 270, 490, 290],
    // 3 : [700, 270, 490, 290],
    // 4: [600, 270, 490, 290],
    // 5: [650, 270, 490, 290],
    // 6 : [775, 270, 490, 290],
    // 7 : [100, 270, 490, 290],
    // 8 : [60, 270, 490, 290],
    // 9 : [200, 270, 490, 290],
    // 10 : [300, 270, 490, 290],
  }

  const layerIndexDict = {
    'input': 0,
    'embedding_Embedding1': 1,
    'lstm_LSTM1': 2,
    'dense_Dense1': 3
  }

  const layerLegendDict = {
    0: {local: 'input-legend', module: 'input-legend', global: 'input-legend'},
    1: {local: 'local-legend-0-1', module: 'module-legend-0', global: 'global-legend'},
    2: {local: 'local-legend-0-1', module: 'module-legend-0', global: 'global-legend'},
    3: {local: 'output-legend', module: 'output-legend', global: 'output-legend'}
  }

  const defaultInputContent = 'Please input your review within 100 words.';
  const exampleReviews = {
    'input': defaultInputContent,
    'empty': 'Helpless Waiting... The text here is for testing as rendering 100 words representation will cost much time I will seek to improve the response speed later',
    'positive':
      `die hard mario fan and i loved this game br br this game starts slightly boring but trust me it\'s worth it as soon as you start your hooked the levels are fun and exiting they will hook you OOV your mind turns to mush i\'m not kidding this game is also orchestrated and is beautifully done br br to keep this spoiler free i have to keep my mouth shut about details but please try this game it\'ll be worth it br br story 9 9 action 10 1 it\'s that good OOV 10 attention OOV 10 average 10`,
    'negative':
      `the mother in this movie is reckless with her children to the point of neglect i wish i wasn\'t so angry about her and her actions because i would have otherwise enjoyed the flick what a number she was take my advise and fast forward through everything you see her do until the end also is anyone else getting sick of watching movies that are filmed so dark anymore one can hardly see what is being filmed as an audience we are impossibly involved with the actions on the screen so then why the hell can\'t we have night vision`
  };
  let selectedReview = 'negative';
  let reviewContent;
  let previousSelectedReview = selectedReview;
  let predictor;
  let inputDim;

  let nodeData;
  let selectedNodeIndex = -1;
  let isExitedFromDetailedView = true;
  let isExitedFromCollapse = true;

  // Helper functions
  const selectedScaleLevelChanged = () => {
    if (svg !== undefined) {
      if (!scaleLevelSet.add(selectedScaleLevel)) {
        console.error('Encounter unknown scale level!');
      }

      // Update nodes and legends
      if (selectedScaleLevel != previousSelectedScaleLevel){
        // We can simply redraw all nodes using the new color scale, or we can
        // make it faster by only redraw certian nodes
        let updatingLayerIndexDict = {
          local: {
            module: [1, 2],
            global: [1, 2, 3, 4]
          },
          module: {
            local: [1, 2],
            global: [1, 2, 3, 4]
          },
          global: {
            local: [1, 2, 3, 4],
            module: [1, 2, 3, 4]
          }
        };

        let updatingLayerIndex = updatingLayerIndexDict[
          previousSelectedScaleLevel][selectedScaleLevel];

        updatingLayerIndex.forEach(l => {
          let range = rnnLayerRanges[selectedScaleLevel][l];
          svg.select(`#rnn-layer-group-${l}`)
            .selectAll('.node-image')
            .each((d, i, g) => drawOutputRNN(d, i, g, range));
        });

 
        // Hide previous legend
        svg.selectAll(`.${previousSelectedScaleLevel}-legend`)
          .classed('hidden', true);

        // Show selected legends
        svg.selectAll(`.${selectedScaleLevel}-legend`)
          .classed('hidden', !detailedMode);
      }
      previousSelectedScaleLevel = selectedScaleLevel;
      selectedScaleLevelStore_rnn.set(selectedScaleLevel);
    }
  }

  // update RNN and Interface
  const updateRNNbasedonStoredReview = async () => {
    // isRNNloaded = false;
    console.time('Construct rnn');
    rnn = await predictor.constructNN(exampleReviews[selectedReview], model_lstm);
    console.timeEnd('Construct rnn');
    // isRNNloaded = true;

    let lstm = rnn[rnn.length -2];
    rnn.lstm = lstm;
    // rnn.rawInput = rnn[0];
    // rnn[0] = rnn.nonPadInput;
    rnnStore.set(rnn);
    console.log('rnn layers are: ', rnn);

    reviewArray = predictor.inputArray;
    reviewArrayStore.set(reviewArray);

    updateRNNLayerRanges(inputDim);
    console.log("rnn layer ranges and MinMax are: ", 
      rnnLayerRanges, rnnLayerMinMax);

    updateRNN(); 
  } 

  // clear the content when focus into textarea div
  const focusReviewContent = async() => {
    if (reviewContent === defaultInputContent) {
      reviewContent = '';
    } 
  }

  // update RNN and interface when review content changed by users
  const reviewContentChanged = async ()=>{
      if (reviewContent.trim()===''){
        console.log('The current review content is empty, please write down your review.')
      } else if (reviewContent.trim() === exampleReviews[selectedReview]){
        console.log('The current Review content does not change');
      } else {
        exampleReviews[selectedReview] = reviewContent.trim();
        updateRNNbasedonStoredReview();
      }
  }

  // update RNN and interface when choose other options
  const reviewOptionClicked = async ()=>{
    if (selectedReview === previousSelectedReview) {
      console.log('The current Review Option does not change');           
    } else {
      previousSelectedReview = selectedReview;
      reviewContent = exampleReviews[selectedReview]; 
      console.log('The current Review is: ', selectedReview);
      console.log('The reviewContent is: ', reviewContent);

      if (selectedReview !== 'input'){   
        d3.select('#review-content')
          .attr('contenteditable', 'false'); 
        updateRNNbasedonStoredReview();

      } else {
        d3.select('#review-content')
          .attr('contenteditable', 'true'); 
          if (reviewContent !== defaultInputContent) {
            updateRNNbasedonStoredReview();
          }
      }      
    }
  }

  // handle the event when click the detail button
  const detailedButtonClicked = () => {
    detailedMode = !detailedMode;
    detailedModeStore_rnn.set(detailedMode);

    if (!isInIntermediateView){
      // Show the legend
      svg.selectAll(`.${selectedScaleLevel}-legend`)
        .classed('hidden', !detailedMode);
      
      svg.selectAll('.input-legend').classed('hidden', !detailedMode);
      svg.selectAll('.output-legend').classed('hidden', !detailedMode);
    }
    
    // Switch the layer name
    svg.selectAll('.layer-detailed-label')
      .classed('hidden', !detailedMode);
    
    svg.selectAll('.layer-label')
      .classed('hidden', detailedMode);
  }

  // The order of the if/else statements in this function is very critical
  const emptySpaceClicked = () => {
    // If detail view -> rewind to intermediate view
    if (detailedViewNum !== undefined) {
          // Setting this for testing purposes currently.
      selectedNodeIndex = -1; 
      // User clicks this node again -> rewind
      svg.select(`rect#underneath-gateway-${detailedViewNum}`)
        .style('opacity', 0);
      detailedViewNum = undefined;
    }

    // If sigmoid view -> rewind to dense layer view
    else if (isInSigmoid) {
      svg.select('.sigmoid-symbol')
        .dispatch('click');
    }

    // If intermediate view -> rewind to overview
    else if (isInIntermediateView) {
      let curLayerIndex = layerIndexDict[selectedNode.layerName];
      quitIntermediateView(curLayerIndex, selectedNode.domG, selectedNode.domI);
      d3.select(selectedNode.domG[selectedNode.domI])
        .dispatch('mouseleave');
    }

    // If pool/act detail view -> rewind to overview
    else if (isInActPoolDetailView) {
      quitActPoolDetailView();
    }
  }

  const quitIntermediateView = (curLayerIndex, g, i) => {
    // If it is the sigmoid detail view, quit that view first
    if (isInSigmoid) {
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

      allowsSigmoidAnimationStore_rnn.set(false);
    }
    isInSigmoidStore_rnn.set(false);
    isInIntermediateView = false;

    // Show the legend
    svg.selectAll(`.${selectedScaleLevel}-legend`)
      .classed('hidden', !detailedMode);
    svg.selectAll('.input-legend').classed('hidden', !detailedMode);
    svg.selectAll('.output-legend').classed('hidden', !detailedMode);

    // Recover control panel UI
    disableControl = false;

    // Recover the input layer node's event
    for (let n = 0; n < rnn[curLayerIndex - 1].length; n++) {
      svg.select(`g#layer-${curLayerIndex - 1}-node-${n}`)
        .on('mouseover', nodeMouseOverHandler)
        .on('mouseleave', nodeMouseLeaveHandler)
        .on('click', nodeClickHandler);
    }

    // Clean up the underneath rects
    svg.select('g.underneath')
      .selectAll('rect')
      .remove();
    detailedViewNum = undefined;

    // Highlight the previous layer and this node
    svg.select(`g#rnn-layer-group-${curLayerIndex - 1}`)
      .selectAll('rect.bounding')
      .style('stroke-width', 1);
    
    d3.select(g[i])
      .select('rect.bounding')
      .style('stroke-width', 1);

    // Highlight the labels
    svg.selectAll(`g#layer-label-${curLayerIndex - 1},
      g#layer-detailed-label-${curLayerIndex - 1},
      g#layer-label-${curLayerIndex},
      g#layer-detailed-label-${curLayerIndex}`)
      .style('font-weight', 'normal');

    // Also unclick the node
    // Record the current clicked node
    selectedNode.layerName = '';
    selectedNode.index = -1;
    selectedNode.data = null;
    isExitedFromCollapse = true;

    // Remove the intermediate layer
    let intermediateLayer = svg.select('g.intermediate-layer');

    // Kill the infinite animation loop
    shouldIntermediateAnimateStore_rnn.set(false);

    intermediateLayer.transition('remove')
      .duration(500)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0)
      .on('end', (d, i, g) => { d3.select(g[i]).remove()});
    
    // Remove the output node overlay mask
    svg.selectAll('.overlay-group').remove();
    
    // Remove the overlay rect
    svg.selectAll('g.intermediate-layer-overlay, g.intermediate-layer-annotation')
      .transition('remove')
      .duration(500)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0)
      .on('end', (d, i, g) => {
        svg.selectAll('g.intermediate-layer-overlay, g.intermediate-layer-annotation').remove();
        svg.selectAll('defs.overlay-gradient').remove();
      });
    
    // Recover the layer if we have drdrawn it
    if (needRedraw[0] !== undefined) {
      let redrawRange = rnnLayerRanges[selectedScaleLevel][needRedraw[0]];
      if (needRedraw[1] !== undefined) {
        svg.select(`g#layer-${needRedraw[0]}-node-${needRedraw[1]}`)
          .select('image.node-image')
          .each((d, i, g) => drawOutput(d, i, g, redrawRange));
      } else {
        svg.select(`g#rnn-layer-group-${needRedraw[0]}`)
          .selectAll('image.node-image')
          .each((d, i, g) => drawOutput(d, i, g, redrawRange));
      }
    }
    
    // Move all layers to their original place
    for (let i = 0; i < numLayers; i++) {
      moveLayerX({layerIndex: i, targetX: nodeCoordinate[i][0].x,
        disable:false, delay:500, opacity: 1});
    }

    moveLayerX({layerIndex: numLayers - 1,
      targetX: nodeCoordinate[numLayers - 1][0].x, opacity: 1,
      disable:false, delay:800, onEndFunc: () => {
        // Show all edges on the last moving animation end
        svg.select('g.edge-group')
          .style('visibility', 'visible');
        svg.select('.lstm-annotation')
            .classed('hidden',false);
        // Recover the input annotation
        svg.select('.input-annotation')
            .classed('hidden', false);
      }});
  }

  const quitActPoolDetailView = () => {
    isInActPoolDetailView = false;
    actPoolDetailViewNodeIndex = -1;

    let layerIndex = layerIndexDict[selectedNode.layerName];
    let nodeIndex = selectedNode.index;
    svg.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
      .select('rect.bounding')
      .classed('hidden', true);

    selectedNode.data.inputLinks.forEach(link => {
      let layerIndex = layerIndexDict[link.source.layerName];
      let nodeIndex = link.source.index;
      svg.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
        .select('rect.bounding')
        .classed('hidden', true);
    })

    // Clean up the underneath rects
    svg.select('g.underneath')
      .selectAll('rect')
      .remove();

    // Show all edges
    let unimportantEdges = svg.select('g.edge-group')
      .selectAll('.edge')
      .filter(d => {
        return d.targetLayerIndex !== actPoolDetailViewLayerIndex;
      })
      .style('visibility', null);
    
    // Recover control UI
    disableControl = false;

    // Show legends if in detailed mode
    svg.selectAll(`.${selectedScaleLevel}-legend`)
      .classed('hidden', !detailedMode);
    svg.selectAll('.input-legend').classed('hidden', !detailedMode);
    svg.selectAll('.output-legend').classed('hidden', !detailedMode);

    // Also dehighlight the edge
    let edgeGroup = svg.select('g.rnn-group').select('g.edge-group');
    edgeGroup.selectAll(`path.edge-${layerIndex}-${nodeIndex}`)
      .transition()
      .ease(d3.easeCubicOut)
      .duration(200)
      .style('stroke', edgeInitColor)
      .style('stroke-width', edgeStrokeWidth)
      .style('opacity', edgeOpacity);

    // Remove the overlay rect
    svg.selectAll('g.intermediate-layer-overlay, g.intermediate-layer-annotation')
      .transition('remove')
      .duration(500)
      .ease(d3.easeCubicInOut)
      .style('opacity', 0)
      .on('end', (d, i, g) => {
        svg.selectAll('g.intermediate-layer-overlay, g.intermediate-layer-annotation').remove();
        svg.selectAll('defs.overlay-gradient').remove();
        svg.select('.input-annotation').classed('hidden', false);
      });

    // Turn the fade out nodes back
    svg.select(`g#rnn-layer-group-${layerIndex}`)
      .selectAll('g.node-group')
      .each((sd, si, sg) => {
        d3.select(sg[si])
          .style('pointer-events', 'all');
    });

    svg.select(`g#rnn-layer-group-${layerIndex - 1}`)
      .selectAll('g.node-group')
      .each((sd, si, sg) => {
        // Recover the old events
        d3.select(sg[si])
          .style('pointer-events', 'all')
          .on('mouseover', nodeMouseOverHandler)
          .on('mouseleave', nodeMouseLeaveHandler)
          .on('click', nodeClickHandler);
    });

    // Deselect the node
    selectedNode.layerName = '';
    selectedNode.index = -1;
    selectedNode.data = null;

    actPoolDetailViewLayerIndex = -1;
  } 

  const prepareToEnterIntermediateView = (d, g, i, curLayerIndex) => {
    isInIntermediateView = true;
    // Hide all legends
    svg.selectAll(`.${selectedScaleLevel}-legend`)
      .classed('hidden', true);
    svg.selectAll('.input-legend').classed('hidden', true);
    svg.selectAll('.output-legend').classed('hidden', true);

    // Hide the input annotation
    svg.select('.input-annotation')
      .classed('hidden', true);

    // Hide the lstm annotation
    svg.select('.lstm-annotation')
      .classed('hidden',true)

    // Highlight the previous layer and this node
    svg.select(`g#rnn-layer-group-${curLayerIndex - 1}`)
      .selectAll('rect.bounding')
      .style('stroke-width', 2);
    
    d3.select(g[i])
      .select('rect.bounding')
      .style('stroke-width', 2);
    
    // Disable control panel UI
    // d3.select('#level-select').property('disabled', true);
    // d3.selectAll('.image-container')
    //   .style('cursor', 'not-allowed')
    //   .on('mouseclick', () => {});
    disableControl = true;
    
    // Allow infinite animation loop
    shouldIntermediateAnimateStore_rnn.set(true);

    // Highlight the labels
    svg.selectAll(`g#layer-label-${curLayerIndex - 1},
      g#layer-detailed-label-${curLayerIndex - 1},
      g#layer-label-${curLayerIndex},
      g#layer-detailed-label-${curLayerIndex}`)
      .style('font-weight', '800');
    
    // Register a handler on the svg element so user can click empty space to quit
    // the intermediate view
    d3.select('#rnn-svg')
      .on('click', emptySpaceClicked);
  }

  const enterDetailView = (curLayerIndex, i) => {
    isInActPoolDetailView = true;
    actPoolDetailViewNodeIndex = i;
    actPoolDetailViewLayerIndex = curLayerIndex;

    // Dynamically position the detail view
    let wholeSvg = d3.select('#rnn-svg');
    let svgYMid = +wholeSvg.style('height').replace('px', '') / 2;
    let svgWidth = +wholeSvg.style('width').replace('px', '');
    let detailViewTop = 100 + svgYMid - 260 / 2;

    let posX = 0;
    // maybe 2 for rnn
    if (curLayerIndex > 2) {
      posX = nodeCoordinate[curLayerIndex - 1][0].x + 50;
      posX = posX / 2 - 500 / 2;
    } else {
      posX = (svgWidth - nodeCoordinate[curLayerIndex][0].x - nodeLength) / 2;
      posX = nodeCoordinate[curLayerIndex][0].x + nodeLength + posX - 500 / 2;

    }

    const detailview = document.getElementById('detailview');
    detailview.style.top = `${detailViewTop}px`;
    detailview.style.left = `${posX}px`;
    detailview.style.position = 'absolute';

    // Hide all edges
    let unimportantEdges = svg.select('g.edge-group')
      .selectAll('.edge')
      .filter(d => {
        return d.targetLayerIndex !== curLayerIndex;
      })
      .style('visibility', 'hidden');
    
    // Disable UI
    disableControl = true;
    
    // Hide input annotaitons
    svg.select('.input-annotation')
      .classed('hidden', true);

    // Hide legends
    svg.selectAll(`.${selectedScaleLevel}-legend`)
      .classed('hidden', true);
    svg.selectAll('.input-legend').classed('hidden', true);
    svg.selectAll('.output-legend').classed('hidden', true);
    svg.select(`#${layerLegendDict[curLayerIndex][selectedScaleLevel]}`)
      .classed('hidden', false);

    // Add overlay rects
    let leftX = nodeCoordinate[curLayerIndex - 1][i].x;
    // +5 to cover the detailed mode long label
    let rightStart = nodeCoordinate[curLayerIndex][i].x + nodeLength + 5;
    // embbedingLength for embbedding layer
    if (curLayerIndex ===1){
      rightStart = rightStart -nodeLength + embbedingLength;
    }

    // Compute the left and right overlay rect width
    let rightWidth = width - rightStart - overlayRectOffset / 2;
    let leftWidth = leftX - nodeCoordinate[0][0].x;

    // The overlay rects should be symmetric
    if (rightWidth > leftWidth) {
      let stops = [{offset: '0%', color: 'rgb(250, 250, 250)', opacity: 0.85},
        {offset: '50%', color: 'rgb(250, 250, 250)', opacity: 0.9},
        {offset: '100%', color: 'rgb(250, 250, 250)', opacity: 1}];
      addOverlayGradient('overlay-gradient-right', stops);
      
      let leftEndOpacity = 0.85 + (0.95 - 0.85) * (leftWidth / rightWidth);
      stops = [{offset: '0%', color: 'rgb(250, 250, 250)', opacity: leftEndOpacity},
        {offset: '100%', color: 'rgb(250, 250, 250)', opacity: 0.85}];
      addOverlayGradient('overlay-gradient-left', stops);
    } else {
      let stops = [{offset: '0%', color: 'rgb(250, 250, 250)', opacity: 1},
        {offset: '50%', color: 'rgb(250, 250, 250)', opacity: 0.9},
        {offset: '100%', color: 'rgb(250, 250, 250)', opacity: 0.85}];
      addOverlayGradient('overlay-gradient-left', stops);

      let rightEndOpacity = 0.85 + (0.95 - 0.85) * (rightWidth / leftWidth);
      stops = [{offset: '0%', color: 'rgb(250, 250, 250)', opacity: 0.85},
        {offset: '100%', color: 'rgb(250, 250, 250)', opacity: rightEndOpacity}];
      addOverlayGradient('overlay-gradient-right', stops);
    }
    
    addOverlayRect('overlay-gradient-right',
      rightStart + overlayRectOffset / 2 + 0.5,
      0, rightWidth, height + svgPaddings.top);
    
    addOverlayRect('overlay-gradient-left',
      nodeCoordinate[0][0].x - overlayRectOffset / 2,
      0, leftWidth, height + svgPaddings.top);

    svg.selectAll('rect.overlay')
      .on('click', emptySpaceClicked);
    
    // Add underneath rectangles
    let underGroup = svg.select('g.underneath');
    let padding = 1;
    for (let n = 0; n < rnn[curLayerIndex - 1].length; n++) {
      underGroup.append('rect')
        .attr('class', 'underneath-gateway')
        .attr('id', `underneath-gateway-${n}`)
        .attr('x', nodeCoordinate[curLayerIndex - 1][n].x - padding)
        .attr('y', nodeCoordinate[curLayerIndex - 1][n].y - padding)
        .attr('width', (1 * nodeLength + 1*embbedingLength + hSpaceAroundGap) + 2 * padding)
        .attr('height', inputNodeHeight + 2 * padding)
        .attr('rx', 10)
        .style('fill', 'rgba(160, 160, 160, 0.3)')
        .style('opacity', 0);
      
      // Update the event functions for these two layers
      svg.select(`g#layer-${curLayerIndex - 1}-node-${n}`)
        .style('pointer-events', 'all')
        .style('cursor', 'pointer')
        .on('mouseover', actPoolDetailViewPreNodeMouseOverHandler)
        .on('mouseleave', actPoolDetailViewPreNodeMouseLeaveHandler)
        .on('click', actPoolDetailViewPreNodeClickHandler);
    }
    underGroup.lower();

    // Highlight the selcted pair
    underGroup.select(`#underneath-gateway-${i}`)
      .style('opacity', 1);
  }

  const actPoolDetailViewPreNodeMouseOverHandler = (d, i, g) => {
    // Highlight the edges
    let layerIndex = layerIndexDict[d.layerName];
    let nodeIndex = d.index;
    let edgeGroup = svg.select('g.rnn-group').select('g.edge-group');
    
    edgeGroup.selectAll(`path.edge-${actPoolDetailViewLayerIndex}-${nodeIndex}`)
      .raise()
      .transition()
      .ease(d3.easeCubicInOut)
      .duration(400)
      .style('stroke', edgeHoverColor)
      .style('stroke-width', '1')
      .style('opacity', 1);
    
    // Highlight its border
    d3.select(g[i]).select('rect.bounding')
      .classed('hidden', false);
    
    // Highlight node's pair
    let associatedLayerIndex = layerIndex - 1;
    if (layerIndex === actPoolDetailViewLayerIndex - 1) {
      associatedLayerIndex = layerIndex + 1;
    }

    svg.select(`g#layer-${associatedLayerIndex}-node-${nodeIndex}`)
      .select('rect.bounding')
      .classed('hidden', false);
  }

  const actPoolDetailViewPreNodeMouseLeaveHandler = (d, i, g) => {
    // De-highlight the edges
    let layerIndex = layerIndexDict[d.layerName];
    let nodeIndex = d.index;
    let edgeGroup = svg.select('g.rnn-group').select('g.edge-group');

    edgeGroup.selectAll(`path.edge-${actPoolDetailViewLayerIndex}-${nodeIndex}`)
      .transition()
      .ease(d3.easeCubicOut)
      .duration(200)
      .style('stroke', edgeInitColor)
      .style('stroke-width', edgeStrokeWidth)
      .style('opacity', edgeOpacity);
    
    // De-highlight its border
    d3.select(g[i]).select('rect.bounding')
      .classed('hidden', true);
    
    // De-highlight node's pair
    let associatedLayerIndex = layerIndex - 1;
    if (layerIndex === actPoolDetailViewLayerIndex - 1) {
      associatedLayerIndex = layerIndex + 1;
    }

    svg.select(`g#layer-${associatedLayerIndex}-node-${nodeIndex}`)
      .select('rect.bounding')
      .classed('hidden', true);
  }

  const actPoolDetailViewPreNodeClickHandler = (d, i, g) => {
    let layerIndex = layerIndexDict[d.layerName];
    let nodeIndex = d.index;

    // Click the pre-layer node in detail view has the same effect as clicking
    // the cur-layer node, which is to open a new detail view window
    svg.select(`g#layer-${layerIndex + 1}-node-${nodeIndex}`)
      .node()
      .dispatchEvent(new Event('click'));
  }

  const intermediateNodeMouseOverHandler = (d, i, g) => {
    if (detailedViewNum !== undefined) { return; }
    svg.select(`rect#underneath-gateway-${d.index}`)
      .style('opacity', 1);
  }

  const intermediateNodeMouseLeaveHandler = (d, i, g) => {
    // return;
    if (detailedViewNum !== undefined) { return; }
    svg.select(`rect#underneath-gateway-${d.index}`)
      .style('opacity', 0);
  }

  const intermediateNodeClicked = (d, i, g, selectedI, curLayerIndex) => {
    d3.event.stopPropagation();
    isExitedFromCollapse = false;
    // Use this event to trigger the detailed view
    if (detailedViewNum === d.index) {
      // Setting this for testing purposes currently.
      selectedNodeIndex = -1; 
      // User clicks this node again -> rewind
      detailedViewNum = undefined;
      svg.select(`rect#underneath-gateway-${d.index}`)
        .style('opacity', 0);
    } 
    // We need to show a new detailed view (two cases: if we need to close the
    // old detailed view or not)
    else {
      // Setting this for testing purposes currently.
      selectedNodeIndex = d.index;
      let inputMatrix = d.output;
      let kernelMatrix = d.outputLinks[selectedI].weight;
      // let interMatrix = singleConv(inputMatrix, kernelMatrix);
      let colorScale = layerColorScales.conv;

      // Compute the color range
      let rangePre = rnnLayerRanges[selectedScaleLevel][curLayerIndex - 1];
      let rangeCur = rnnLayerRanges[selectedScaleLevel][curLayerIndex];
      let range = Math.max(rangePre, rangeCur);

      // User triggers a different detailed view
      if (detailedViewNum !== undefined) {
        // Change the underneath highlight
        svg.select(`rect#underneath-gateway-${detailedViewNum}`)
          .style('opacity', 0);
        svg.select(`rect#underneath-gateway-${d.index}`)
          .style('opacity', 1);
      }
      
      // Dynamically position the detail view
      let wholeSvg = d3.select('#rnn-svg');
      let svgYMid = +wholeSvg.style('height').replace('px', '') / 2;
      let svgWidth = +wholeSvg.style('width').replace('px', '');
      let detailViewTop = 100 + svgYMid - 250 / 2;
      let positionX = intermediateLayerPosition[Object.keys(layerIndexDict)[curLayerIndex]];

      let posX = 0;
      if (curLayerIndex > 6) {
        posX = (positionX - svgPaddings.left) / 2;
        posX = svgPaddings.left + posX - 486 / 2;
      } else {
        posX = (svgWidth + svgPaddings.right - positionX) / 2;
        posX = positionX + posX - 486 / 2;
      }

      const detailview = document.getElementById('detailview');
      detailview.style.top = `${detailViewTop}px`;
      detailview.style.left = `${posX}px`;
      detailview.style.position = 'absolute';

      detailedViewNum = d.index;

      // Send the currently used color range to detailed view
      nodeData.colorRange = range;
      nodeData.inputIsInputLayer = curLayerIndex <= 1;
    }
  }

  const nodeClickHandler = (d, i, g) => {
    d3.event.stopPropagation();
    let nodeIndex = d.index;

    // Record the current clicked node
    selectedNode.layerName = d.layerName;
    selectedNode.index = d.index;
    selectedNode.data = d;
    selectedNode.domI = i;
    selectedNode.domG = g;

    // Record data for detailed view.
    if (d.type ==='embedding'||d.type === 'lstm'|| d.type === 'dense') {
      let data = [];
      for (let j = 0; j < d.inputLinks.length; j++) {
        data.push({
          input: d.inputLinks[j].source.output,
          kernel: d.inputLinks[j].weight,
          output: d.inputLinks[j].dest.output,
        })
      }
      let curLayerIndex = layerIndexDict[d.layerName];
      data.colorRange = rnnLayerRanges[selectedScaleLevel][curLayerIndex];
      data.isInputInputLayer = curLayerIndex <= 1;
      nodeData = data;
    }

    let curLayerIndex = layerIndexDict[d.layerName];

    if (d.type === 'embedding') {
      isExitedFromDetailedView = false;
      if (!isInActPoolDetailView) {
        // Enter the act pool detail view
        enterDetailView(curLayerIndex, d.index);
      } else {
        if (d.index === actPoolDetailViewNodeIndex) {
          // Quit the act pool detail view
          quitActPoolDetailView();
        } 
        else {
          // Switch the detail view input to the new clicked pair

          // Remove the previous selection effect
          svg.select(`g#layer-${curLayerIndex}-node-${actPoolDetailViewNodeIndex}`)
            .select('rect.bounding')
            .classed('hidden', true);

          svg.select(`g#layer-${curLayerIndex - 1}-node-${actPoolDetailViewNodeIndex}`)
            .select('rect.bounding')
            .classed('hidden', true);
          
          let edgeGroup = svg.select('g.rnn-group').select('g.edge-group');
      
          edgeGroup.selectAll(`path.edge-${curLayerIndex}-${actPoolDetailViewNodeIndex}`)
            .transition()
            .ease(d3.easeCubicOut)
            .duration(200)
            .style('stroke', edgeInitColor)
            .style('stroke-width', edgeStrokeWidth)
            .style('opacity', edgeOpacity);
          
          let underGroup = svg.select('g.underneath');
          underGroup.select(`#underneath-gateway-${actPoolDetailViewNodeIndex}`)
            .style('opacity', 0);
        
          // Add selection effect on the new selected pair
          svg.select(`g#layer-${curLayerIndex}-node-${nodeIndex}`)
            .select('rect.bounding')
            .classed('hidden', false);

          svg.select(`g#layer-${curLayerIndex - 1}-node-${nodeIndex}`)
            .select('rect.bounding')
            .classed('hidden', false);

          edgeGroup.selectAll(`path.edge-${curLayerIndex}-${nodeIndex}`)
            .raise()
            .transition()
            .ease(d3.easeCubicInOut)
            .duration(400)
            .style('stroke', edgeHoverColor)
            .style('stroke-width', '1')
            .style('opacity', 1);

          underGroup.select(`#underneath-gateway-${nodeIndex}`)
            .style('opacity', 1);

          actPoolDetailViewNodeIndex = nodeIndex;
        }
      }
    }

    // Enter the second view (layer-view) when user clicks a conv node
    if ((d.type === 'lstm' || d.layerName === 'dense_Dense1') && !isInIntermediateView) {
      prepareToEnterIntermediateView(d, g, nodeIndex, curLayerIndex);

      if (d.layerName === 'dense_Dense1'){
        drawDense(curLayerIndex, d, nodeIndex, width, height,
          intermediateNodeMouseOverHandler, intermediateNodeMouseLeaveHandler,
          intermediateNodeClicked);
      }
      else if (d.layerName === 'lstm_LSTM1'){
        // todo: expand the view on time scale
        drawLstm(curLayerIndex, d, nodeIndex, width, height,
          intermediateNodeMouseOverHandler, intermediateNodeMouseLeaveHandler,
          intermediateNodeClicked);
      }
    }
    // Quit the layerview
    else if ((d.type === 'lstm' || d.layerName === 'dense_Dense1') && isInIntermediateView) {
      quitIntermediateView(curLayerIndex, g, i);
    }
  }

  const nodeMouseOverHandler = (d, i, g) => {
    // if (isInIntermediateView || isInActPoolDetailView) { return; }
    if (isInIntermediateView) { return; }

    // Highlight the edgesr
    let layerIndex = layerIndexDict[d.layerName];
    let nodeIndex = d.index;
    let edgeGroup = svg.select('g.rnn-group').select('g.edge-group');
    
    edgeGroup.selectAll(`path.edge-${layerIndex}-${nodeIndex}`)
      .raise()
      .transition()
      .ease(d3.easeCubicInOut)
      .duration(400)
      .style('stroke', edgeHoverColor)
      .style('stroke-width', '1')
      .style('opacity', 1);
    
    // Highlight its border
    d3.select(g[i]).select('rect.bounding')
      .classed('hidden', false);
    
    // Highlight source's border
    if (d.inputLinks.length === 1) {
      let link = d.inputLinks[0];
      let layerIndex = layerIndexDict[link.source.layerName];
      let nodeIndex = link.source.index;
      svg.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
        .select('rect.bounding')
        .classed('hidden', false);
    } else {
      svg.select(`g#rnn-layer-group-${layerIndex - 1}`)
        .selectAll('g.node-group')
        .selectAll('rect.bounding')
        .classed('hidden', false);
    }

    // Highlight the output text
    if (d.layerName === '"dense_Dense1"') {
      d3.select(g[i])
        .select('.output-text')
        .style('opacity', 0.8)
        .style('text-decoration', 'underline');
    }

    /* Use the following commented code if we have non-linear model
    d.inputLinks.forEach(link => {
      let layerIndex = layerIndexDict[link.source.layerName];
      let nodeIndex = link.source.index;
      svg.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
        .select('rect.bounding')
        .classed('hidden', false);
    });
    */
  }

  const nodeMouseLeaveHandler = (d, i, g) => {
    // Screenshot
    // return;

    if (isInIntermediateView) { return; }
    
    // Keep the highlight if user has clicked
    if (isInActPoolDetailView || (
      d.layerName !== selectedNode.layerName ||
      d.index !== selectedNode.index)){
      let layerIndex = layerIndexDict[d.layerName];
      let nodeIndex = d.index;
      let edgeGroup = svg.select('g.rnn-group').select('g.edge-group');
      
      edgeGroup.selectAll(`path.edge-${layerIndex}-${nodeIndex}`)
        .transition()
        .ease(d3.easeCubicOut)
        .duration(200)
        .style('stroke', edgeInitColor)
        .style('stroke-width',  d =>d.targetLayerIndex ===2 ? edgeStrokeWidth:edgeStrokeWidth*4)
        .style('opacity', edgeOpacity);
      
      if (d.type !== 'dense') {
        d3.select(g[i]).select('rect.bounding').classed('hidden', true);
      }

      if (d.inputLinks.length === 1) {
        let link = d.inputLinks[0];
        let layerIndex = layerIndexDict[link.source.layerName];
        let nodeIndex = link.source.index;
        svg.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
          .select('rect.bounding')
          .classed('hidden', true);
      } else {
        svg.select(`g#rnn-layer-group-${layerIndex - 1}`)
          .selectAll('g.node-group')
          .selectAll('rect.bounding')
          .classed('hidden', d => d.layerName !== selectedNode.layerName ||
            d.index !== selectedNode.index);
      }

      // Dehighlight the output text
      if (d.layerName === 'dense_Dense1') {
        d3.select(g[i])
          .select('.output-text')
          .style('fill', 'black')
          .style('opacity', 0.5)
          .style('text-decoration', 'none');
      }

      /* Use the following commented code if we have non-linear model
      d.inputLinks.forEach(link => {
        let layerIndex = layerIndexDict[link.source.layerName];
        let nodeIndex = link.source.index;
        hoverInfoStore_rnn.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
          .select('rect.bounding')
          .classed('hidden', true);
      });
      */
    }
  }

  const directPredict = (input, model) => {
    if (predictor) {
      // await predictor.predictResult(input, model).
            // then(res => result = res);

      let result = predictor.predictResult(input, model);

      console.log('Direct Result: Inference result (0 - negative; 1 - positive): ' +
                    result.score.toFixed(6) +
                  ' (elapsed: ' + result.elapsed.toFixed(2) + ' ms)');
    } else {
      console.log('something went wrong with predictor');
    }

  }

  onMount(async () => {
      // Create RNN
      console.log(`-----------Creating RNN---------------`);
      overview = d3.select(rnnOverviewComponent);
      width = Number(overview.style('width').replace('px', '')) -
        svgPaddings.left - svgPaddings.right;

      wholeSvg_rnn = d3.select(rnnOverviewComponent)
        .select('#rnn-svg');
      svg = wholeSvg_rnn.append('g')
        .attr('class','main-svg')
        .attr('transform',`translate(${svgPaddings.left}, 0)`);

      svgStore_rnn.set(svg);

      // width = Number(wholeSvg_rnn.style('width').replace('px', '')) -
      //   svgPaddings.left - svgPaddings.right;
      height = Number(wholeSvg_rnn.style('height').replace('px', '')) -
        svgPaddings.top - svgPaddings.bottom;

      let rnnGroup = svg.append('g')
        .attr('class','rnn-group');
      
      let underGroup_rnn = svg.append('g')
        .attr('class', 'underneath');

      let svgYMid_rnn = +wholeSvg_rnn.style('height').replace('px','') / 2;

      detailedViewAbsCoords = {
        1 : [600, 100 + svgYMid_rnn - 220 / 2, 490, 290],
        2: [500, 100 + svgYMid_rnn - 220 / 2, 490, 290],
        // 3 : [700, 100 + svgYMid - 220 / 2, 490, 290],
        // 4: [600, 100 + svgYMid - 220 / 2, 490, 290],
        // 5: [650, 100 + svgYMid - 220 / 2, 490, 290],
        // 6 : [850, 100 + svgYMid - 220 / 2, 490, 290],
        // 7 : [100, 100 + svgYMid - 220 / 2, 490, 290],
        // 8 : [60, 100 + svgYMid - 220 / 2, 490, 290],
        // 9 : [200, 100 + svgYMid - 220 / 2, 490, 290],
        // 10 : [300, 100 + svgYMid - 220 / 2, 490, 290],
      }

      // Define global arrow marker end
      svg.append("defs")
        .append("marker")
        .attr("id", 'marker')
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 6)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .style('stroke-width', 1.2)
        .style('fill', 'gray')
        .style('stroke', 'gray')
        .attr("d", "M0,-5L10,0L0,5");

      // Alternative arrow head style for non-interactive annotation
      svg.append("defs")
        .append("marker")
        .attr("id", 'marker-alt')
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 6)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .append("path")
        .style('fill', 'none')
        .style('stroke', 'gray')
        .style('stroke-width', 2)
        .attr("d", "M-5,-10L10,0L-5,10");

      predictor = await new SentimentPredictor().init(LOCAL_URLS);
      model_lstm = predictor.model;
      console.log("The rnn model is: ", model_lstm);

      // check the result first
      directPredict(`${exampleReviews[selectedReview]}`, model_lstm);

      console.time('Construct rnn');
      rnn = await predictor.constructNN(`${exampleReviews[selectedReview]}`, model_lstm);
      console.timeEnd('Construct rnn');
      // isRNNloaded = true;
      // // Ignore the rawInput layer for now, because too many <pad> node in input layer will 
      // // cause the exploration of edges, which will cost performance loss in interface
      // rnn.rawInput = rnn[0];
      // rnn[0] = rnn.nonPadInput;
      let lstm = rnn[rnn.length -2];
      rnn.lstm = lstm;
      rnnStore.set(rnn);
      console.log('rnn layers are: ', rnn);

      reviewArray = predictor.inputArray;
      reviewArrayStore.set(reviewArray);

      inputDim = model_lstm.layers[0].inputDim;
      updateRNNLayerRanges(inputDim);
      console.log("rnn layer ranges and MinMax are: ", 
        rnnLayerRanges, rnnLayerMinMax);

      // Create and draw the RNN view
      drawRNN(width, height, rnnGroup, nodeMouseOverHandler, 
      nodeMouseLeaveHandler, nodeClickHandler);
  });


  function handleExitFromDetiledConvView(event) {
    if (event.detail.text) {
      detailedViewNum = undefined;
      svg.select(`rect#underneath-gateway-${selectedNodeIndex}`)
        .style('opacity', 0);
      selectedNodeIndex = -1; 
    }
  }

  function handleExitFromDetiledEmbeddingView(event) {
    if (event.detail.text) {
      quitActPoolDetailView();
      isExitedFromDetailedView = true;
    }
  }

  function handleExitFromDetiledSigmoidView(event) {
    sigmoidDetailViewInfo.show = false;
    sigmoidDetailViewStore_rnn.set(sigmoidDetailViewInfo);
  }
</script>

<style>
  .rnnOverview {
    padding: 0;
    height: 100%;
    width: 100%;
    display: flex;
    position: relative;
    flex-direction: column;
    justify-content: space-between;
    align-items: flex-start;
  }

  .control-container {
    padding: 5px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
  }

  .right-control {
    display: flex;
  }

  .left-control {
    display: flex;
    align-items: center;
  }

  .control > .select > #level-select {
    padding-left: 2em;
    padding-right: 2em;
  }

  #example-select {
    padding-left: 2em;
    padding-right: 2em;
  }

  .textarea {
    width:400px;
    min-height:48px;
    max-height:240;
    _height:120px;
    margin-left:auto;
    margin-right:auto;
    outline:0;
    border:1pxsolid#a0b3d6;
    font-size:12px;
    line-height:24px;
    padding:2px;
    word-wrap:break-word;
    overflow-x:hidden;
    overflow-y:auto;

    border-color:rgba(82,168,236,0.8);
    box-shadow:inset01px3pxrgba(0,0,0,0.1),008pxrgba(82,168,236,0.6);
  }

  .rnn {
    width: 100%;
    padding: 0;
    background: var(--light-gray);
    display: flex;
  }

  .spinner-item {
	  min-width: 250px;
	  min-height: 250px;
	  display: flex;
	  justify-content: center;
	  align-items: center;
    position: absolute;
    top: 50%;
    left:50%;
    transform:translate(-50%,-50%);
  }

  svg {
    margin: 0 auto;
    min-height: 500px;
    max-height: 1200px;
    height: calc(100vh - 100px);
    width: 100vw;
    display:flex
  }

  .is-very-small {
    font-size: 12px;
  }

  #detailed-button {
    margin-right: 10px;
    color: #dbdbdb;
    transition: border-color 300ms ease-in-out, color 200ms ease-in-out;
  }

  #detailed-button.is-activated, #detailed-button.is-activated:hover {
    color: #3273dc;
    border-color: #3273dc;
  }

  #detailed-button:hover {
    color: #b5b5b5;
  }

  #hover-label {
    transition: opacity 300ms ease-in-out;
    text-overflow: ellipsis;
    pointer-events: none;
    margin-left: 5px;
  }

  .review {
    padding: 5px 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
  }

  #review-text {
    font-size: 100%;
    width: 600%;
    height: 50px;
  }

  :global(canvas) {
    image-rendering: crisp-edges;
  }

  :global(.layer-label, .layer-detailed-label, .layer-intermediate-label) {
    font-size: 12px;
    opacity: 0.8;
    text-anchor: middle;
  }

  :global(.colorLegend) {
    font-size: 10px;
  }

  :global(.legend) {
    transition: opacity 400ms ease-in-out;
  }

  :global(.legend > rect) {
    opacity: 1;
  }

  :global(.legend text, .legend line, .legend path) {
    opacity: 0.7;
  }

  :global(.legend#output-legend > rect) {
    opacity: 1;
  }

  :global(.hidden) {
    opacity: 0;
    pointer-events: none;
  }

  :global(.very-strong) {
    stroke-width: 3px;
  }

  :global(.bounding, .edge, .edge-group, foreignObject, .bounding-flatten,
    .underneath-gateway, .input-annotation) {
    transition: opacity 300ms ease-in-out;
  }

  :global(rect.bounding) {
    transition: stroke-width 800ms ease-in-out, opacity 300ms ease-in-out;
  }

  :global(.annotation-text) {
    pointer-events: none;
    font-size: 10px;
    font-style: italic;
    fill: gray;
  }

  /* Change the cursor style on the detailed view input and output matrices */
  :global(rect.square) {
    cursor: crosshair;
  }

  :global(.animation-control-button) {
    font-family: FontAwesome;
    opacity: 0.8;
    cursor: pointer;
  }

</style>

<div class="rnnOverview"
  bind:this={rnnOverviewComponent}>

  <div class="control-container">

    <div class="left-control">  

      <div class="control is-very-small has-icons-left"
        title="Change input using different examples">
          <span class="icon is-left">
            <i class="fas fa-palette"></i>
          </span>

          <div class="select">
            <select bind:value="{selectedReview}"
              disabled={disableControl}
              id="example-select" 
              class="form-control" 
              on:change= "{disableControl ? '' : reviewOptionClicked}">
              <!-- <option value="empty"> Please choose one example</option> -->
              <option value="positive">Positive example</option>
              <option value="negative">Negative example</option>
              <option value="input">Input your review</option>
            </select>
          </div>      
      </div>

      <button class="button is-very-small is-link is-light"
        id="hover-label"
        style="opacity:{hoverInfo_rnn.show ? 1 : 0}">
        <span class="icon" style="margin-right: 5px;">
          <i class="fas fa-crosshairs "></i>
        </span>
        <span id="hover-label-text">
          {hoverInfo_rnn.text}
        </span>
      </button>

    </div>

    <div class="right-control">

      <button class="button is-very-small"
        id="detailed-button"
        disabled={disableControl}
        class:is-activated={detailedMode}
        on:click={detailedButtonClicked}>
        <span class="icon">
          <i class="fas fa-eye"></i>
        </span>
        <span id="hover-label-text">
          Show detail
        </span>
      </button>

      <div class="control is-very-small has-icons-left"
        title="Change color scale range">
        <span class="icon is-left">
          <i class="fas fa-layer-group"></i>
        </span>

        <div class="select">
          <select bind:value={selectedScaleLevel} id="level-select"
            disabled={disableControl}>
            <option value="local">Unit</option>
            <option value="module">Module</option>
            <option value="global">Global</option>
          </select>
        </div>
      </div>

    </div>
    
  </div>

  <div class="review">
    <div 
      bind:textContent={reviewContent} 
      class="textarea" id="review-content" 
      disabled={disableControl}
      contenteditable="false" 
      on:focus="{focusReviewContent}"
      on:blur="{reviewContentChanged}">
        {exampleReviews[selectedReview]}
    </div>
    <!-- <textarea id="review-text" on:blur="{disableControl ? '' : reviewOptionClicked}" maxlength = "100o">{exampleReviews[selectedReview]}</textarea> -->
  </div>

  <div class="rnn" id="rnnView">
    {#if  !rnn}
      <div class="spinner-item" title="Jumper">
        <Jumper size="120" color="#0abab5" />
      </div>
    {/if}
    <svg id="rnn-svg"></svg>
  </div>

</div>

<div id='detailview'>
  {#if selectedNode.data && selectedNode.data.type === 'embedding'}
    <EmbeddingView on:message={handleExitFromDetiledEmbeddingView} input={[nodeData[0].input]} 
                    kernel={nodeData[0].kernel} output={[nodeData[0].output]}
                    dataRange={nodeData.colorRange}
                    isExited={isExitedFromDetailedView}/>
  {:else if sigmoidDetailViewInfo.show}
    <SigmoidView logits={sigmoidDetailViewInfo.logits}
                 logitColors={sigmoidDetailViewInfo.logitColors}
                 selectedI={sigmoidDetailViewInfo.selectedI}
                 highlightI={sigmoidDetailViewInfo.highlightI}
                 outputName={sigmoidDetailViewInfo.outputName}
                 outputValue={sigmoidDetailViewInfo.outputValue}
                 startAnimation={sigmoidDetailViewInfo.startAnimation}
                 on:xClicked={handleExitFromDetiledSigmoidView}
                 on:mouseOver={sigmoidDetailViewMouseOverHandler}
                 on:mouseLeave={sigmoidDetailViewMouseLeaveHandler}/>
  {/if}
</div>