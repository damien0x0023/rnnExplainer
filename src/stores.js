import { writable } from 'svelte/store';

export const cnnStore = writable([]);

export const svgStore = writable(undefined);

export const vSpaceAroundGapStore = writable(undefined);
export const hSpaceAroundGapStore = writable(undefined);

export const nodeCoordinateStore = writable([]);
export const selectedScaleLevelStore = writable(undefined);

export const cnnLayerRangesStore = writable({});
export const cnnLayerMinMaxStore = writable([]);

export const needRedrawStore = writable([undefined, undefined]);

export const detailedModeStore = writable(true);

export const shouldIntermediateAnimateStore = writable(false);

export const isInSoftmaxStore = writable(false);
export const softmaxDetailViewStore = writable({});
export const allowsSoftmaxAnimationStore = writable(false);

export const hoverInfoStore = writable({});

export const modalStore = writable({});

export const intermediateLayerPositionStore = writable({});

// add Store for rnn
export const rnnStore = writable([]);

export const svgStore_rnn = writable(undefined);

export const vSpaceAroundGapStore_rnn = writable(undefined);
export const hSpaceAroundGapStore_rnn = writable(undefined);

export const nodeCoordinateStore_rnn = writable([]);
export const selectedScaleLevelStore_rnn = writable(undefined);

export const rnnLayerRangesStore = writable({});
export const rnnLayerMinMaxStore = writable([]);

export const needRedrawStore_rnn = writable([undefined, undefined]);

export const detailedModeStore_rnn = writable(true);

export const shouldIntermediateAnimateStore_rnn = writable(false);

//may not use in the rnn
export const isInSoftmaxStore_rnn = writable(false);
export const softmaxDetailViewStore_rnn = writable({});
export const allowsSoftmaxAnimationStore_rnn = writable(false);

export const hoverInfoStore_rnn = writable({});

export const modalStore_rnn = writable({});

export const intermediateLayerPositionStore_rnn = writable({});