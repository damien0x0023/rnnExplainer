<script>
  import { createEventDispatcher } from 'svelte';
  import { array1d, getMatrixSliceFromOutputHighlights,
    getVisualizationSizeConstraint, getRNNMatrixSliceFromHighlights, gridData
  } from './DetailviewUtils.js';
  import Dataview from './Dataview.svelte';
  import DataviewRNN from './DataviewRNN.svelte';

  export let image;
  // export let kernel;
  export let output;
  export let isPaused;
  export let dataRange;

  const dispatch = createEventDispatcher();
  const padding = 0;
  let dimIndex;
  let padded_input_size = image.length + padding * 2;
  $: padded_input_size = image.length + padding * 2;

  let gridInputMatrixSlice = gridData([[0]]);
  let gridOutputMatrixSlice = gridData([[0]]);
  let inputHighlights = array1d(image.length * image[0].length, (i) => true);
  let outputHighlights = array1d(output.length * output[0].length, (i) => true);
  let interval;
  $ : {
    let inputHighlights = array1d(image.length * image[0].length, (i) => true);
    let outputHighlights = array1d(output.length * output[0].length, (i) => true);
    let interval;
  }

  let counter;

  // lots of replication between mouseover and start-relu. TODO: fix this.
  function startEmbedding() {
    counter = 0;
    if (interval) clearInterval(interval);
    interval = setInterval(() => {
      if (isPaused) return;
      const flat_animated = counter % (output.length * output[0].length);
      outputHighlights = array1d(output.length * output[0].length, (i) => false);
      // inputHighlights = array1d(image.length * image[0].length, (i) => undefined);
      const animatedH = Math.floor(flat_animated / output[0].length);
      const animatedW = flat_animated % output[0].length;
      dimIndex = animatedH * output[0].length + animatedW
      outputHighlights[dimIndex] = true;
      // inputHighlights[animatedH * image[0].length] = true;
      // const inputMatrixSlice = getRNNMatrixSliceFromHighlights(image, inputHighlights, 1);
      // gridInputMatrixSlice = gridData(inputMatrixSlice);
      const outputMatrixSlice = getRNNMatrixSliceFromHighlights(output, outputHighlights);
      gridOutputMatrixSlice = gridData(outputMatrixSlice);
      counter++;
    }, 250)
  }

  function handleMouseover(event) {
    outputHighlights = array1d(output.length * output[0].length, (i) => false);
    const animatedH = event.detail.hoverH;
    const animatedW = event.detail.hoverW;
    dimIndex = animatedH * output[0].length + animatedW;
    outputHighlights[dimIndex] = true;
    // inputHighlights = array1d(image.length * image[0].length, (i) => undefined);
    // inputHighlights[animatedH * image[0].length +animatedW] = true;
    // const inputMatrixSlice = getRNNMatrixSliceFromHighlights(image, inputHighlights, 1);
    // gridInputMatrixSlice = gridData(inputMatrixSlice);
    const outputMatrixSlice = getRNNMatrixSliceFromHighlights(output, outputHighlights);
    gridOutputMatrixSlice = gridData(outputMatrixSlice);
    isPaused = true;
    dispatch('message', {
      text: isPaused
    });
  }

  startEmbedding();
  let gridImage = gridData(image);
  // let gridEmbed = gridData(kernel);
  let gridOutput = gridData(output, getVisualizationSizeConstraint(output[0].length));
  $ : {
    startEmbedding();
    gridImage = gridData(image)
    // gridEmbed = gridData(kernel);
    gridOutput = gridData(output, getVisualizationSizeConstraint(output[0].length));
  }
</script>

<style>
  .column {
    padding: 5px;
  }
</style>

<div class="column has-text-centered">
  <div class="header-text">
    Token 
    <!-- ({image.length}, {image[0].length}) -->
  </div>
  <DataviewRNN on:message={handleMouseover} data={gridImage} highlights={inputHighlights} outputLength={output.length}
      isKernelMath={true} constraint={getVisualizationSizeConstraint(image.length)} dataRange={dataRange} stride={1}/>  
</div>
<div class="column has-text-centered">
  <span>
    Vocab[{image[0][0]}][{dimIndex+1}]
    =
    <DataviewRNN data={gridOutputMatrixSlice} highlights={outputHighlights} isKernelMath={true} 
      constraint={20} dataRange={dataRange}/>
  </span> 
</div>
<div class="column has-text-centered">
  <div class="header-text">
    Output ({output.length}, {output[0].length})
  </div>
  <DataviewRNN on:message={handleMouseover} data={gridOutput} highlights={outputHighlights} isKernelMath={false} 
      outputLength={output[0].length} constraint={getVisualizationSizeConstraint(output[0].length)*3} dataRange={dataRange} stride={1}/>
</div>