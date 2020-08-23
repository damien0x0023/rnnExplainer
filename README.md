# RNN Explainer

An interactive visualization application designed to help non-experts learn about Recurrent Neural Networks (RNNs). This version is editted by CNN Explainer from Poloclub. The following is the sources:

[![Build Status](https://travis-ci.com/poloclub/cnn-explainer.svg?branch=master)](https://travis-ci.com/poloclub/cnn-explainer)
[![arxiv badge](https://img.shields.io/badge/arXiv-2004.15004-red)](http://arxiv.org/abs/2004.15004)

<!-- <a href="https://youtu.be/HnWIHWFbuUQ" target="_blank"><img src="https://i.imgur.com/TIKlgt6.png" style="max-width:100%;"></a> -->

For more information, check out their manuscript:

[**CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization**](https://arxiv.org/abs/2004.15004).
Wang, Zijie J., Robert Turko, Omar Shaikh, Haekyu Park, Nilaksh Das, Fred Hohman, Minsuk Kahng, and Duen Horng Chau.
arXiv preprint 2020. arXiv:2004.15004.


## Live Demo

For a live demo of their App, visit: http://poloclub.github.io/cnn-explainer/

## Running Locally

Clone or download this repository:

```
git clone git@github.com:damien0x0023/rnnExplainer.git

# use degit if you don't want to download commit histories
degit damien0x0023/rnnExplainer
```

Install the dependencies:

```
npm install 
```
or
```
yarn
```

Then run RNN Explainer:

```
npm run dev
```
or
```
yarn dev
```

Navigate to [localhost:5000](https://localhost:5000). You should see the Explainer running in your broswer.

To see how we trained the CNN or RNN, visit the directory [`./tiny-vgg/`](tiny-vgg) or [`./imdb/`](imdb).

## Credits


## License
The software is available under the [MIT License](https://github.com/poloclub/cnn-explainer/blob/master/LICENSE).


