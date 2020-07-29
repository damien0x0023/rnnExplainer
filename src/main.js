import App from './App.svelte';

const app = new App({
	target: document.body,
	props: { 
		defaultNN: 'RNN',
	}
});

export default app;