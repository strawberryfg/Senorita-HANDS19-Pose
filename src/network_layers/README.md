# Custom Network Layers

[**Spoiler**] The code is not intuituive or easy to use whatsoever, and I apologize for this. This is due, in some part to the framework and the language, which I cannot control. If you are seeking for a solution that produces state-of-the-art results while at the same time being easily comprehended in 10 minutes, consider closing this page.

You may find these three repos [c2f](https://github.com/strawberryfg/c2f-3dhm-human-caffe), [intreg](https://github.com/strawberryfg/int-3dhuman-I1), [DeepModel](https://github.com/strawberryfg/DeepModel_hand) useful to have a sense of how custom layers are defined and deployed in the framework. This is *potentially more difficult* than PyTorch, Keras.


## Installation
For the time being I am using this [framework](https://github.com/happynear/caffe-windows) by happynear, as you see I have only access to a Windows laptop. But it should work fine under Ubuntu too.

To configure my personalized layers, you will have to add **hpp** files and **new cpp** files. See the above repos or [this page](https://github.com/BVLC/caffe/wiki/Development) to get an idea.

# Functionalities

----

## DeepHandModelGetHands19ChaDepth
``` 
Output the revised bounding box of depth patch, 3D cube bounding box, and 2D image center coordinate (pixel location of center of mass).
``` 

## "DeepHandModelPinholeCameraOrigin"
``` 
Projecting real-world 3D coordinates to raw depth image.
``` 