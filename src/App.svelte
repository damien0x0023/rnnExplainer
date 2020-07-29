<script>
	export let defaultNN;
	import Explainer from './Explainer.svelte';
	import Header from './Header.svelte';

	const NNlist= [{ 
			id: 1, 
			name: `Convolutional Neural Network`, 
			alis: `CNN`,
		},{ 
			id: 2, 
			name: `Recurrent Neural Network`, 
			alis: `RNN`,
		}
	];

	let selected;
	
	selected = defaultNN && defaultNN===`RNN` ? NNlist[1]: NNlist[0];

</script>

<style>
</style>

<div id="app-page">
	<Header bind:selectedNN={selected} NNs={NNlist}/> 
	<Explainer nn={selected.alis}/>
	<!-- {#if nn==='cnn'}
	<Explainer />
	{:else if nn==='rnn'}
	<RNNExplainer />
	{/if} -->
</div>
