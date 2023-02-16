# Practicing transfer learning

## Introduction

This file is a step by step tutorial to practice basics of transfer learning on an industrial dataset composed of pictures, for a use case of quality control. You will also find the dataset and Jupyter notebooks with code that could help you address the questions.

## What is transfer learning
- Take a pretrained model
- Freeze weights
- Remove last layer
- Replace it by what makes sense to your question
- Train only last layer

## Cast defect dataset

- This dataset provides image data of impellers for submersible pumps.
- There are many types of defect in casting like blow holes, pinholes, burr, shrinkage defects, mould material defects, pouring metal defects, metallurgical defects, etc.
- Raw dataset = 519 ok + 781 defect 
- Augmented dataset = 7348 pictures

## Classic CV approach
- SIFT (or AKAZE)
- Bag of words
- Classification with SVM
- See `defects_AKAZE_SVM.ipynb`

## Transfer learning approach
- Take a pretrained image classification model
- Use transfer learning
- Let’s do it with Keras
- https://keras.io/guides/transfer_learning/
- See `defects_transfer.ipynb`

## What tweaks to try
- With or without transfer learning
- With different kinds of image classification models
- With or without data augmentation
- With a different number of input pictures

## Ideas to go further
- Explainability
