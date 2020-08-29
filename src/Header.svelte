<script>
	// let NNs= [
	// 	{id: 1, name: `Convolutional Neural Network`, alis: `CNN`},
	// 	{id: 2, name: `Recurrent Neural Network`, alis: `RNN`}
	// ];

	// export  let defaultNN;
	// let selected = defaultNN && defaultNN==='rnn'?NNs[1]: NNs[0];

	// function handleChanged(){
	// 	alert(`answer question ${selected.id} (${selected.alis}) with "${selected.name}"`);
	// }

	export let NNs;
	export let selectedNN;
	

</script>

<style>
	#header {
		height: 40px;
		display: flex;
		padding: 0 20px;
		align-items: stretch;
		background: rgb(10, 162, 186);
    	justify-content: space-between;
	}

	#logo {
		width: 100%;
		display: flex;
		align-items: center;
	}

	#logo-text {
		font-size: 25px;
		color: var(--light-gray);
		font-family: 'A Love of Thunder';
		dominant-baseline: baseline;
		margin-right: 10px;
		text-align: left;
        border: 1px dotted black;
        width: 25%;

	}

	.logo-svg{
		width:50%;
		height:35px;
		text-align: center;
        border: 1px dotted black;
	}

	#svg-logo-tagline {
		font-size: 22px;
		fill: rgb(255, 255, 255);
		text-align: center;
		width: 50%;
		font-family: 'Neucha';
		position:absolute;
		bottom:0;
	}

	.icons {
		display: flex;
		justify-content: flex-start;
		align-items: center;
	}

	.icon {
		width: 27px;
		height: 27px;
		margin-left: 15px;
	}

	.icon a{
		width: 100%;
		height: 100%;
	}

	.icon img {
    width: 100%;
    height: 100%;
    object-fit: contain;
	}

	.is-very-small {
    font-size: 12px;
  	}

	.control {
        text-align: right;
        border: 1px dotted black;
        width: 25%;
	}
	.control > .select > #NN-select {
    padding-left: 1em;
    padding-right: 0.5em;
  	}

  	.NNselector {
		/* display: flex; */
		/* justify-content: flex-start; */
		/* align-items: center; */
	    display: flex;
		position: relative;
		flex-direction: column;
		justify-content: space-between;
		align-items: flex-start;
	}
	
</style>

<div id="header">
	<div id="logo">
		<div id="logo-text">
			{selectedNN.alis} Explainer
    	</div>

		<svg class="logo-svg" >
			<defs>
				<filter x="0%" y="0%" width="100%" height="100%" filterUnits="objectBoundingBox" id="chalk-texture">
					<feTurbulence type="fractalNoise" baseFrequency="2" numOctaves="5" stitchTiles="stitch" result="f1">
					</feTurbulence>
					<feColorMatrix type="matrix" values="0 0 0 0 0, 0 0 0 0 0, 0 0 0 0 0, 0 0 0 -1.5 1.5" result="f2">
					</feColorMatrix>
					<feComposite operator="in" in2="f2" in="SourceGraphic" result="f3">
					</feComposite>
				</filter>
			</defs>

			<g class="svg-logo" filter="url(#chalk-texture)" transform="translate(0,30)">				
				<text id="svg-logo-tagline">
					Learn {selectedNN.name} ({selectedNN.alis}) in your browser!
				</text>
			</g>
		</svg>
		
		<div class = "control is-very-small has-icons-left" title="Change NN">
			<span class="icon is-right">
				<i class="fas fa-bezier-curve"></i>
			</span>
	
			<div class = "select">
				<div id="NN-select">
					<select bind:value={selectedNN} class="NNselector">
							{#each NNs as NN}
								<option value={NN}>
									{NN.alis}
								</option>
							{/each}
					</select>
				</div>
			</div>
	
		</div>
  	</div>



	<!-- <p>{selectedNN.alis}</p> -->

	<!-- <div class="icons">
	
		<div class="icon" title="Research paper">
			<a target="_blank" href="http://zijie.wang/redirect/cnn-explainer/">
				<img src="PUBLIC_URL/assets/img/pdf.png" alt="pdf icon"/>
			</a>
		</div>

		<div class="icon" title="Demo video">
			<a target="_blank" href="https://youtu.be/HnWIHWFbuUQ">
				<img src="PUBLIC_URL/assets/img/youtube.png" alt="youtube icon"/>
			</a>
		</div>

		<div class="icon" title="Open-source code">
			<a target="_blank" href="https://github.com/poloclub/cnn-explainer">
				<img src="PUBLIC_URL/assets/img/github.png" alt="github icon"/>
			</a>
		</div>
	</div> -->
</div>