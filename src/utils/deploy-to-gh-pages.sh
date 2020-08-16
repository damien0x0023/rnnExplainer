#!/bin/bash
set -o errexit

# config
git config --global user.email "jyh91517@gmail.com"
git config --global user.name "damien0x0023"

# build
git clone git@github.com:damien0x0023/rnnExplainer.git
cd rnnExplainer

yarn
yarn build

mkdir dist
copy -r ./public/* ./dist
sed -i 's/\/assets/\/cnn-explainer\/assets/g' ./dist/index.html

git add dist
git commit -m "Deploy gh-pages from Travis"
git subtree push --prefix dist origin gh-pages
