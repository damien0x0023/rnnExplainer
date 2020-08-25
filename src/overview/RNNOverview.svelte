<script>
  // Svelte functions
  import { onMount } from 'svelte';
  import { 
    rnnStore, rnnLayerMinMaxStore, rnnLayerRangesStore, 
    svgStore_rnn, vSpaceAroundGapStore_rnn, hSpaceAroundGapStore_rnn,
    nodeCoordinateStore_rnn, selectedScaleLevelStore_rnn, needRedrawStore_rnn,
    detailedModeStore_rnn, shouldIntermediateAnimateStore_rnn, isInSoftmaxStore_rnn, 
    softmaxDetailViewStore_rnn, hoverInfoStore_rnn, allowsSoftmaxAnimationStore_rnn, 
    modalStore_rnn, intermediateLayerPositionStore_rnn
  } from '../stores.js';

  // Svelte views
  import ConvolutionView from '../detail-view/Convolutionview.svelte';
  import ActivationView from '../detail-view/Activationview.svelte';
  import PoolView from '../detail-view/Poolview.svelte';
  import SoftmaxView from '../detail-view/Softmaxview.svelte';
  import Modal from './Modal.svelte'
  import ArticleRNN from '../article/ArticleRNN.svelte';

  const HOSTED_URLS = {
    model:
        'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/model.json',
    metadata:
        'https://storage.googleapis.com/tfjs-models/tfjs/sentiment_cnn_v1/metadata.json'
  };

  const LOCAL_URLS = {
    model: 'PUBLIC_URL/resources/model.json',
    metadata: 'PUBLIC_URL/resources/metadata.json'
  };

  // Overview functions
  import { loadTrainedModel_rnn, constructRNN, SentimentPredictor } from '../utils/rnn-tf.js';
  // import { loadTrainedModel, constructCNN } from '../utils/cnn-tf.js';
  import { 
    // overviewConfig, 
    rnnOverviewConfig } from '../config.js';

  import {
    addOverlayRect, drawConv1, drawConv2, drawConv3, drawConv4
  } from './intermediate-draw.js';

  import {
    moveLayerX, addOverlayGradient
  } from './intermediate-utils.js';

  import {
    drawFlatten, softmaxDetailViewMouseOverHandler, softmaxDetailViewMouseLeaveHandler
  } from './flatten-draw.js';

  // import {
  //   drawOutput, drawCNN, updateCNN, updateCNNLayerRanges, drawCustomImage
  // } from './overview-draw.js';

  import {
    drawOutputRNN, drawRNN, updateRNN, updateRNNLayerRanges, drawCustomReivew
  } from './overview-drawRNN.js';

  // View bindings
  let rnnOverviewComponent;
  let scaleLevelSet = new Set(['local', 'module', 'global']);
  let selectedScaleLevel = 'local';
  selectedScaleLevelStore_rnn.set(selectedScaleLevel);
  let previousSelectedScaleLevel = selectedScaleLevel;
  let wholeSvg_rnn = undefined;
  let svg_rnn = undefined;

  // let wholeSvg_cnn = undefined;
  // let svg_cnn = undefined;

  $: selectedScaleLevel, selectedScaleLevelChanged();

  // Configs
  const layerColorScales = rnnOverviewConfig.layerColorScales;
  const nodeLength = rnnOverviewConfig.nodeLength;
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
  let needRedraw_rnn = [undefined, undefined];
  needRedrawStore_rnn.subscribe( value => {needRedraw_rnn = value;} );

  let nodeCoordinate_rnn = undefined;
  nodeCoordinateStore_rnn.subscribe( value => {nodeCoordinate_rnn = value;} )

  let rnnLayerRanges = undefined;
  rnnLayerRangesStore.subscribe( value => {rnnLayerRanges = value;} )

  let rnnLayerMinMax = undefined;
  rnnLayerMinMaxStore.subscribe( value => {rnnLayerMinMax = value;} )

  let detailedMode_rnn = undefined;
  detailedModeStore_rnn.subscribe( value => {detailedMode_rnn = value;} )

  let shouldIntermediateAnimate_rnn = undefined;
  shouldIntermediateAnimateStore_rnn.subscribe(value => {
    shouldIntermediateAnimate_rnn = value;
  })

  let vSpaceAroundGap_rnn = undefined;
  vSpaceAroundGapStore_rnn.subscribe( value => {vSpaceAroundGap_rnn = value;} )

  let hSpaceAroundGap_rnn = undefined;
  hSpaceAroundGapStore_rnn.subscribe( value => {hSpaceAroundGap_rnn = value;} )

  let modalInfo_rnn = undefined;
  modalStore_rnn.subscribe( value => {modalInfo_rnn = value;} )

  let hoverInfo_rnn = undefined;
  hoverInfoStore_rnn.subscribe( value => {hoverInfo_rnn = value;} )

  let intermediateLayerPosition_rnn = undefined;
  intermediateLayerPositionStore_rnn.subscribe ( value => {
    intermediateLayerPosition_rnn = value;} )

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
  // let cnn = undefined;
  let rnn = undefined;

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

  let imageOptions = [
    {file: 'boat_1.jpeg', class: 'lifeboat'},
    {file: 'bug_1.jpeg', class: 'ladybug'},
    {file: 'pizza_1.jpeg', class: 'pizza'},
    {file: 'pepper_1.jpeg', class: 'bell pepper'},
    {file: 'bus_1.jpeg', class: 'bus'},
    {file: 'koala_1.jpeg', class: 'koala'},
    {file: 'espresso_1.jpeg', class: 'espresso'},
    {file: 'panda_1.jpeg', class: 'red panda'},
    {file: 'orange_1.jpeg', class: 'orange'},
    {file: 'car_1.jpeg', class: 'sport car'}
  ];
  let selectedImage = imageOptions[1].file;

  let selectedReview;
  const exampleReviews = {
  'empty': 'Helpless Waiting... The text here is for testing as rendering 100 words representation will cost much time I will seek to improve the responce speed later',
  'positive':
      `die hard mario fan and i loved this game br br this game starts slightly boring but trust me it\'s worth it as soon as you start your hooked the levels are fun and exiting they will hook you OOV your mind turns to mush i\'m not kidding this game is also orchestrated and is beautifully done br br to keep this spoiler free i have to keep my mouth shut about details but please try this game it\'ll be worth it br br story 9 9 action 10 1 it\'s that good OOV 10 attention OOV 10 average 10`,
  'negative':
      `the mother in this movie is reckless with her children to the point of neglect i wish i wasn\'t so angry about her and her actions because i would have otherwise enjoyed the flick what a number she was take my advise and fast forward through everything you see her do until the end also is anyone else getting sick of watching movies that are filmed so dark anymore one can hardly see what is being filmed as an audience we are impossibly involved with the actions on the screen so then why the hell can\'t we have night vision`
  };

  let nodeData;
  let selectedNodeIndex = -1;
  let isExitedFromDetailedView = true;
  let isExitedFromCollapse = true;
  let customImageURL = null;

  // Helper functions
  const selectedScaleLevelChanged = () => {
    if (svg_rnn !== undefined) {
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
          svg_rnn.select(`#rnn-layer-group-${l}`)
            .selectAll('.node-image')
            .each((d, i, g) => drawOutputRNN(d, i, g, range));
        });

 
        // Hide previous legend
        svg_rnn.selectAll(`.${previousSelectedScaleLevel}-legend`)
          .classed('hidden', true);

        // Show selected legends
        svg_rnn.selectAll(`.${selectedScaleLevel}-legend`)
          .classed('hidden', !detailedMode_rnn);
      }
      previousSelectedScaleLevel = selectedScaleLevel;
      selectedScaleLevelStore_rnn.set(selectedScaleLevel);
    }
  }

  // responce to click other image, todo: change image to change review option
  const imageOptionClicked = async (e) => {
    // todo: need to rewrite for the content of review
    let newImageName = d3.select(e.target).attr('data-imageName');

    if (newImageName !== selectedImage) {
      selectedImage = newImageName;

      // Re-compute the CNN using the new input image
      // rnn = await constructCNN(`PUBLIC_URL/assets/img/${selectedImage}`, model);
      rnn = await constructRNN(`${exampleReviews[selectedReview]}`, 
        LOCAL_URLS.metadata, model_lstm);
      // Ignore the meanless input <pad> for now
      // let orignalInput = rnn[0];
      // rnn[0].filter(d => d.output !== 0);
      // rnn.rawInput = orignalInput;
      // rnnStore.set(rnn);

      // Update all scales used in the CNN view
      updateRNNLayerRanges();
      updateRNN();
    }
  }

  // responce to click the custom Image Button
  const customImageClicked = () => {

    // Case 1: there is no custom image -> show the modal to get user input
    if (customImageURL === null) {
      modalInfo_rnn.show = true;
      modalInfo_rnn.preImage = selectedImage;
      modalStore_rnn.set(modalInfo_rnn);
    }

    // Case 2: there is an existing custom image, not the focus -> switch to this image
    else if (selectedImage !== 'custom') {
      let fakeEvent = {detail: {url: customImageURL}};
      handleCustomImage(fakeEvent);
    }

    // Case 3: there is an existing custom image, and its the focus -> let user
    // upload a new image
    else {
      modalInfo_rnn.show = true;
      modalInfo_rnn.preImage = selectedImage;
      modalStore_rnn.set(modalInfo_rnn);
    }

    if (selectedImage !== 'custom') {
      selectedImage = 'custom';
    }

  }

  // cannot load the image to cannot load the text
  const handleModalCanceled = (event) => {
    // User cancels the modal without a successful image, so we restore the
    // previous selected image as input
    selectedImage = event.detail.preImage;
  }

  // change custom image to custom review
  const handleCustomImage = async (event) => {
    // User gives a valid image URL
    customImageURL = event.detail.url;

    // Re-compute the CNN using the new input image
    // rnn = await constructCNN(customImageURL, model);
    rnn = await constructRNN(`${exampleReviews[selectedReview]}`, 
        LOCAL_URLS.metadata, model_lstm);
    // Ignore the flatten layer for now
    // let flatten = cnn[cnn.length - 2];
    // cnn.splice(cnn.length - 2, 1);
    // cnn.flatten = flatten;
    rnnStore.set(rnn);

    // Update the UI
    let customImageSlot = d3.select(rnnOverviewComponent)
      .select('.custom-image').node();
    drawCustomReivew(customImageSlot, rnn[0]);

    // Update all scales used in the RNN view
    updateRNNLayerRanges();
    updateRNN();
  }

  // handle the event when click the detail button
  const detailedButtonClicked = () => {
    detailedMode_rnn = !detailedMode_rnn;
    detailedModeStore_rnn.set(detailedMode_rnn);

    if (!isInIntermediateView){
      // Show the legend
      svg_rnn.selectAll(`.${selectedScaleLevel}-legend`)
        .classed('hidden', !detailedMode_rnn);
      
      svg_rnn.selectAll('.input-legend').classed('hidden', !detailedMode_rnn);
      svg_rnn.selectAll('.output-legend').classed('hidden', !detailedMode_rnn);
    }
    
    // Switch the layer name
    svg_rnn.selectAll('.layer-detailed-label')
      .classed('hidden', !detailedMode_rnn);
    
    svg_rnn.selectAll('.layer-label')
      .classed('hidden', detailedMode_rnn);
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
    if (d.type === 'conv' || d.type === 'relu' || d.type === 'pool') {
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

    if (d.type == 'relu' || d.type == 'pool') {
      isExitedFromDetailedView = false;
      if (!isInActPoolDetailView) {
        // Enter the act pool detail view
        enterDetailView(curLayerIndex, d.index);
      } else {
        if (d.index === actPoolDetailViewNodeIndex) {
          // Quit the act pool detail view
          quitActPoolDetailView();
        } else {
          // Switch the detail view input to the new clicked pair

          // Remove the previous selection effect
          svg_rnn.select(`g#layer-${curLayerIndex}-node-${actPoolDetailViewNodeIndex}`)
            .select('rect.bounding')
            .classed('hidden', true);

          svg_rnn.select(`g#layer-${curLayerIndex - 1}-node-${actPoolDetailViewNodeIndex}`)
            .select('rect.bounding')
            .classed('hidden', true);
          
          let edgeGroup = svg_rnn.select('g.rnn-group').select('g.edge-group');
      
          edgeGroup.selectAll(`path.edge-${curLayerIndex}-${actPoolDetailViewNodeIndex}`)
            .transition()
            .ease(d3.easeCubicOut)
            .duration(200)
            .style('stroke', edgeInitColor)
            .style('stroke-width', edgeStrokeWidth)
            .style('opacity', edgeOpacity);
          
          let underGroup = svg_rnn.select('g.underneath');
          underGroup.select(`#underneath-gateway-${actPoolDetailViewNodeIndex}`)
            .style('opacity', 0);
        
          // Add selection effect on the new selected pair
          svg_rnn.select(`g#layer-${curLayerIndex}-node-${nodeIndex}`)
            .select('rect.bounding')
            .classed('hidden', false);

          svg_rnn.select(`g#layer-${curLayerIndex - 1}-node-${nodeIndex}`)
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
    if ((d.type === 'conv' || d.layerName === 'dense_Dense1') && !isInIntermediateView) {
      prepareToEnterIntermediateView(d, g, nodeIndex, curLayerIndex);

      if (d.layerName === 'conv_1_1') {
        drawConv1(curLayerIndex, d, nodeIndex, width, height,
          intermediateNodeMouseOverHandler, intermediateNodeMouseLeaveHandler,
          intermediateNodeClicked);
      }

      else if (d.layerName === 'conv_1_2') {
        drawConv2(curLayerIndex, d, nodeIndex, width, height,
          intermediateNodeMouseOverHandler, intermediateNodeMouseLeaveHandler,
          intermediateNodeClicked);
      }

      else if (d.layerName === 'conv_2_1') {
        drawConv3(curLayerIndex, d, nodeIndex, width, height,
          intermediateNodeMouseOverHandler, intermediateNodeMouseLeaveHandler,
          intermediateNodeClicked);
      }
      
      else if (d.layerName === 'conv_2_2') {
        drawConv4(curLayerIndex, d, nodeIndex, width, height,
          intermediateNodeMouseOverHandler, intermediateNodeMouseLeaveHandler,
          intermediateNodeClicked);
      }
    
      else if (d.layerName === 'dense_Dense1') {
        drawFlatten(curLayerIndex, d, nodeIndex, width, height);
      }
    }
    // Quit the layerview
    else if ((d.type === 'conv' || d.layerName === 'dense_Dense1') && isInIntermediateView) {
      quitIntermediateView(curLayerIndex, g, i);
    }
  }

  const nodeMouseOverHandler = (d, i, g) => {
    // if (isInIntermediateView || isInActPoolDetailView) { return; }
    if (isInIntermediateView) { return; }

    // Highlight the edgesr
    let layerIndex = layerIndexDict[d.layerName];
    let nodeIndex = d.index;
    let edgeGroup = svg_rnn.select('g.rnn-group').select('g.edge-group');
    
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
      svg_rnn.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
        .select('rect.bounding')
        .classed('hidden', false);
    } else {
      svg_rnn.select(`g#rnn-layer-group-${layerIndex - 1}`)
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
      svg_rnn.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
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
      let edgeGroup = svg_rnn.select('g.rnn-group').select('g.edge-group');
      
      edgeGroup.selectAll(`path.edge-${layerIndex}-${nodeIndex}`)
        .transition()
        .ease(d3.easeCubicOut)
        .duration(200)
        .style('stroke', edgeInitColor)
        .style('stroke-width', edgeStrokeWidth)
        .style('opacity', edgeOpacity);

      d3.select(g[i]).select('rect.bounding').classed('hidden', true);

      if (d.inputLinks.length === 1) {
        let link = d.inputLinks[0];
        let layerIndex = layerIndexDict[link.source.layerName];
        let nodeIndex = link.source.index;
        svg_rnn.select(`g#layer-${layerIndex}-node-${nodeIndex}`)
          .select('rect.bounding')
          .classed('hidden', true);
      } else {
        svg_rnn.select(`g#rnn-layer-group-${layerIndex - 1}`)
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

  onMount(async () => {
    // Create RNN
    console.log(`-----------Creating RNN---------------`);
    wholeSvg_rnn = d3.select(rnnOverviewComponent)
      .select('#rnn-svg');
    svg_rnn = wholeSvg_rnn.append('g')
      .attr('class','main-svg')
      .attr('transform',`translate(${svgPaddings.left}, 0)`);

    svgStore_rnn.set(svg_rnn);

    width = Number(wholeSvg_rnn.style('width').replace('px', '')) -
      svgPaddings.left - svgPaddings.right;
    height = Number(wholeSvg_rnn.style('height').replace('px', '')) -
      svgPaddings.top - svgPaddings.bottom;

    let rnnGroup = svg_rnn.append('g')
      .attr('class','rnn-group');
    
    let underGroup_rnn = svg_rnn.append('g')
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
    svg_rnn.append("defs")
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
    svg_rnn.append("defs")
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

    // model_lstm = await loadTrainedModel_rnn(LOCAL_URLS.model);
    // console.log("The rnn model is: ",model_lstm);

    let predictor = await new SentimentPredictor().init(LOCAL_URLS);
    model_lstm = predictor.model;
    console.log("The rnn model is: ", model_lstm);

    let result;
    await predictor.predictResult(`${exampleReviews[selectedReview]}`, model_lstm).
          then(res => result = res);
    console.log('Direct Result: Inference result (0 - negative; 1 - positive): ' +
                  result.score.toFixed(6) +
                ' (elapsed: ' + result.elapsed.toFixed(2) + ' ms)');

    console.time('Construct rnn');
    // rnn = await constructRNN(`${exampleReviews[selectedReview]}`, 
    //   LOCAL_URLS.metadata, model_lstm);
    rnn = await predictor.constructNN(`${exampleReviews[selectedReview]}`, model_lstm);
    console.timeEnd('Construct rnn');

    let inputArray = predictor.inputArray;
    console.log('input text array is: ', inputArray);

    // Ignore the rawInput layer for now, because too many <pad> node in input layer will 
    // cause the exploration of edges, which will cost performance loss in interface
    rnn.rawInput = rnn[0];
    rnn[0] = rnn.nonPadInput;
    rnnStore.set(rnn);

    console.log('rnn layers are: ', rnn);

    let inputDim = model_lstm.layers[0].inputDim;
    updateRNNLayerRanges(inputDim);
    console.log("rnn layer ranges and MinMax are: ", 
      rnnLayerRanges, rnnLayerMinMax);

    // Create and draw the RNN view
    // drawRNN(width, height, rnnGroup, nodeMouseOverHandler, 
    // nodeMouseLeaveHandler, nodeClickHandler);
    drawRNN(width, height, rnnGroup, nodeMouseOverHandler, nodeMouseLeaveHandler, null, inputArray);

    });

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

  .cnn {
    width: 100%;
    padding: 0;
    background: var(--light-gray);
    display: flex;
  }

  .rnn {
    width: 100%;
    padding: 0;
    background: var(--light-gray);
    display: flex;
  }

  svg {
    margin: 0 auto;
    min-height: 490px;
    max-height: 700px;
    height: calc(100vh - 100px);
    width: 100vw;
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

  .image-container {
    width: 40px;
    height: 40px;
    border-radius: 4px;
    display: inline-block;
    position: relative;
    border: 2.5px solid #1E1E1E;
    margin-right: 10px;
    cursor: pointer;
    pointer-events: all;
    transition: border 300ms ease-in-out;
  }

  .image-container img {
    object-fit: cover;
    max-width: 100%;
    max-height: 100%;
    z-index: -1;
    transition: opacity 300ms ease-in-out;
  }

  .image-container.inactive {
    border: 2.5px solid rgb(220, 220, 220);
  }

  .image-container.inactive > img {
    opacity: 0.3;
  }

  .image-container.inactive:hover > img {
    opacity: 0.6;
  }

  .image-container.inactive.disabled {
    border: 2.5px solid rgb(220, 220, 220);
    cursor: not-allowed;
  }

  .image-container.inactive.disabled:hover {
    border: 2.5px solid rgb(220, 220, 220);
    cursor: not-allowed;
  }

  .image-container.inactive.disabled > img {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .image-container.inactive.disabled:hover > img {
    opacity: 0.3;
    cursor: not-allowed;
  }

  .image-container.inactive > .edit-icon {
    color: #BABABA;
  }

  .image-container.inactive:hover > .edit-icon {
    color: #777777;
  }

  .image-container.inactive:hover {
    border: 2.5px solid #1E1E1E;
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

  .edit-icon {
    position: absolute;
    bottom: -6px;
    right: -7px;
    font-size: 7px;
    color: #1E1E1E;
    transition: color 300ms ease-in-out;
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
      <!-- {#each imageOptions as image, i}
        <div class="image-container"
          on:click={disableControl ? () => {} : imageOptionClicked}
          class:inactive={selectedImage !== image.file}
          class:disabled={disableControl}
          data-imageName={image.file}>
          <img src="PUBLIC_URL/assets/img/{image.file}"
            alt="image option"
            title="{image.class}"
            data-imageName={image.file}/>
        </div>
      {/each} -->

      <!-- <div class="image-container"
          class:inactive={selectedImage !== 'custom'}
          class:disabled={disableControl}
          data-imageName={'custom'}
          on:click={disableControl ? () => {} : customImageClicked}>

          <img class="custom-image"
            src="PUBLIC_URL/assets/img/plus.svg"
            alt="plus button"
            title="Add new input image"
            data-imageName="custom"/>

          <span class="fa-stack edit-icon"
            class:hidden={customImageURL === null}>
            <i class="fas fa-circle fa-stack-2x"></i>
            <i class="fas fa-pen fa-stack-1x fa-inverse"></i>
          </span>

      </div> -->

      <!-- <button class="button is-very-small is-link is-light"
        id="hover-label"
        style="opacity:{hoverInfo_rnn.show ? 1 : 0}">
        <span class="icon" style="margin-right: 5px;">
          <i class="fas fa-crosshairs "></i>
        </span>
        <span id="hover-label-text">
          {hoverInfo_rnn.text}
        </span>
      </button> -->

      <select bind:value={selectedReview} id="test-example-select" class="form-control">
        <!-- <option value="empty"> Please choose one example</option> -->
        <!-- <option value="positive">Positive example</option> -->
        <option value="negative">Negative example</option>
      </select>
    </div>

    <div class="right-control">

      <button class="button is-very-small"
        id="detailed-button"
        disabled={disableControl}
        class:is-activated={detailedMode_rnn}
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
          <i class="fas fa-palette"></i>
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
    <textarea id="review-text">{exampleReviews[selectedReview]}</textarea>
  </div>

  <div class="ui"></div>

  <div class="rnn" id="rnnView">
    <span id='ui' class="status"></span>
    <svg id="rnn-svg"></svg>
  </div>

  <!-- <div class="cnn">
    <svg id="cnn-svg"></svg>
  </div> -->

</div>

<!-- <ArticleRNN/> -->

<div id='detailview'>
 
</div>

<Modal on:xClicked={handleModalCanceled}
  on:urlTyped={handleCustomImage}/>