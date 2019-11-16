# Own implementation of PointNet
Under development! Expecting completion around 12/2019.

This project is a personal implementation of the object classification portion of PointNet [1] and is implemented in TensorFlow 2.0, including utilization of `tf.keras` layers and `tf.data.Dataset`. An accompanying white paper overview can be found [here](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a).

## Setup
Used PCL. If you don't have it and are using a Mac, I recommend installing via Homebrew. Final model downloaded at ...

## Dataset

<div align="center">
  <p><img src="figs/hist.svg"></p>
  <p>Fig. 1: Given two input frames, the images represented <br/>by (c) and (d) are passed as inputs to the CNN.</p>
</div>

## Model Architecture
The original PointNet architecture is shown below in Fig. 2. This project implements the classification portion, but adding the segmentation portion is 

<div align="center">
  <p><img src="figs/architecture.png"></p>
  <p>Fig. 2: PointNet model architecture.</p>
</div>

## Training
With ModelNet40 located in the project root directory, training can be launched by running `python3 src/train.py`. Optional command-line arguments include batch size, number of epochs, initial learning rate, and whether to use wandb to monitor training.

Training consisted of experimenting with constant vs exponential decay learning rate, learning rate and batch norm momentum warm-up, and anchor loss [2]. Somewhat surprisingly, anchor loss ... 

## Inference
As mentioned in the setup section, the final model can be downloaded at ... With the model checkpoints in the `model/` directory, one can perform inference by running `python3 ???`. Optional command-line arguments include pointing to a different checkpoint and visualizing the point cloud contained in `<file>`.

## Results

## References
[1] [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation, C. Qi et al., 2016](https://arxiv.org/abs/1612.00593)
[2] [Anchor Loss: Modulating Loss Scale based on Prediction Difficulty, S. You et al., 2019](https://arxiv.org/abs/1909.11155)
