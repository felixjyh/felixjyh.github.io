# <center>Assignment 3 Part 2 Animal Classification (Report) </center>

<style>
.center 
{
  width: 600px;
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

<style>
.centertable 
{
  display: table;
  margin-left: auto;
  margin-right: auto;
}
</style>

## Introduction

In this part, we build a network model based on fastai library to realize the classification of 10 kinds of animals. We select appropriate models, propose effective solutions and give corresponding experimental results and explanations.

## Method description and experimental results

The hardware we used was a personal computer with i5-10400 CPU and 4060ti GPU. After many experiments and tests, for the dataset animal10, the loss will no longer decrease after about 15 epoch training of the model. In order to ensure the model is effectively fine-tuned, we set the epoch to 30. In this case, due to hardware limitations, especially speed and GPU memory limitations, in order to ensure that the loss of the model can be trained within 6 hours to no longer decline, that is, after training 30 epochs, we choose the resnet26d model with fewer parameters.

<div class="center">

![alt text](images/image.png)

</div>

We chose the animal10 dataset provided by kaggle, where the training set contained 20,944 images and the validation set contained 5,235 images. Above are some examples of the data set. The data set contains 'butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep',
'spider', 'squirrel' these 10 categories.


In previous experiments on paddy's disease detection, we learned that augmentation was effective, so in this experiment, we also configured appropriate augmentation transforms in our DataBlock. In order to make the best use of the GPU and ensure faster training speed, we set the batch size to 32 after debugging.

<div class="center">

![alt text](images/image-1.png)

</div>

The above figure is the classification result of some images achieved by the model.

We look at the error examples when the model classifies some pictures, and make a quantitative analysis of the error cases. Some examples of incorrect classification are shown in the figure below.

<div class="center">

![alt text](images/image-3.png)

</div>

The confusion matrix is shown in the following figure.

<div class="center">

![alt text](images/image-2.png)

</div>

It can be seen that most of the error samples are similar species of animals. For example, among the butterfly samples with the most errors, 10 of them are classified as spiders. Of course, these two species are both insects. While the 6 samples of sheep were identified as dogs, but they were never identified as spiders. There are similarities between sheep and dogs, but not spiders.

In order to improve the classification accuracy, we can appropriately increase the depth of the model. In theory, the deeper the model, the better its ability to fit, the better its ability to distinguish similar species, and ultimately, the higher the classification accuracy. On resnet26d, we debugged and finally achieved an error rate of 0.2.