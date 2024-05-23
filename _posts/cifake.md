# <center>Assignment 3 Part 3 CIFAKE (Report) </center>

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

Recent research on CIFAKE classification, focusing on distinguishing between real and AI-generated images, has leveraged advanced architectures to address the growing challenge of synthetic image detection. The CIFAKE dataset[1], created by combining real images from the CIFAR-10 dataset and synthetic images generated via latent diffusion models, serves as the foundation for these studies.

In this part, we try to classify real vs fake images and give appropriate discussion and approach. 

<div class="center">

![alt text](images/image.png)

</div>

## Method description and experimental results

To solve the problem of cifake differentiation, we use fastai library to build and compare different deep learning networks, and investigate the current cutting-edge methods. Neural networks are the most suitable for classifying pictures. We try to copy and modify other people's methods.[2][4] Below are examples of some cifake data sets.


<div class="center">

![alt text](images/image-4.png)

</div>

We trained resnet26d, vgg19_bn[3], densenet121 and other models to for experiments. 

In each experiment, we used the lr_find function to find the appropriate learning rate, as shown in the figure below.


<div class="center">

![alt text](images/image-1.png)

</div>

Among all the models, the final experiment with the highest classification accuracy is the deepest densenet121, which achieved 98.41% accuracy. Below are some examples of classification results.

<div class="center">

![alt text](images/image-2.png)

</div>

The confusion matrix was used to analyze the results. As shown in the figure below, it can be seen that many real pictures were identified as fake pictures, out of 254 in total, only 64 fake pictures were identified as real. Therefore, on the cifake data set, the model could easily identify real pictures incorrectly.

<div class="center">

![alt text](images/image-3.png)

</div>

## Reference

[1] Bird J J, Lotfi A. Cifake: Image classification and explainable identification of ai-generated synthetic images[J]. IEEE Access, 2024.

[2] Bartos G E, Akyol S. Deep Learning for Image Authentication: A Comparative Study on Real and AI-Generated Image Classification[J].

[3] H. V, K. P and M. A, "Art of Detection: Custom CNN and VGG19 for Accurate Real Vs Fake Image Identification," 2023 6th International Conference on Recent Trends in Advance Computing (ICRTAC), Chennai, India, 2023, pp. 306-312, doi: 10.1109/ICRTAC59277.2023.10480775.

[4] https://www.kaggle.com/code/archietram/test-for-jet