# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization_barchat_trainingset.png "Visualization Bar Chat of the Test Image Set"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/examples/visualization_oneofeach.png "One Image for each Label"
[image10]: ./examples/image.png "Original Keep Right Image"
[image11]: ./examples/image_gs.png "Grayscaled Keep Right Image"
[image12]: ./examples/image_norm.png "Normalized Keep Right Image"
[image13]: ./fromweb/3.jpg "Speed limit (60km/h)"
[image14]: ./fromweb/19.jpg "Dangerous curve to the left"
[image15]: ./fromweb/23.jpg "Slippery road"
[image16]: ./fromweb/25.jpg "Road work"
[image17]: ./fromweb/38.jpg "Keep right"
[image18]: ./fromweb/39.jpg "Keep left"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/amilendra/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is : 34799
* The size of the validation set is : 4410
* The size of test set is : 12630
* The shape of a traffic sign image is : (32, 32, 3)
* The number of unique classes/labels in the data set is : 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data in the training data set is spread across each of the 43 labels. We can see that the number of images is not uniformly distributed over the image label. Some images have over 2000 training images but some have less than 250. This should affect the accurary when predicting images with lesser training data.

![alt text][image1]

I plotted one image for each label(unsorted) to get an idea of what type of signs are we trying to predict. 
![alt text][image9]

My first impression is the quality of the images are very poor, with blurred focusing, extremely low lighting conditions making it difficult to classity even to a human who did not know what the images are.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because rather than color, most of the information in the images are in the intensity of the light.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image10]
![alt text][image11]

As a last step, I normalized the image data because some images clearly have low intensity, so averaging intensities of the images bring out the feature more clearly.

![alt text][image10]
![alt text][image12]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Convolution 1x1	    | 2x2 stride, valid padding, outputs 1x1x400	|
| RELU					|												|
| Fully connected		| input 400, output 120        					|
| RELU					|												|
| Fully connected		| input 120, output 84        					|
| RELU					|												|
| Fully connected		| input 84, output 43        					|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a LeNet architecture.
I used the same architecture as the one given as an example for analysing the MNIST database. Other than changing the number of output classes from 10 to 43, other parameters were left unchanged.

I used an AdamOptimizer with a learning rate of 0.001 which was also used in the example. The epochs used was 30(original was 20) while the batch size was left unchanged at 128.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I increased the number of epochs because that ensured the validation accuracy remained stable above the 0.93 accuracy that was required in the assignment.

At first I only added a grayscale transformation for the preprocessing step, but that did not improve results much. Looking at some sample images made me realize that normalization is required because of the poor lighting conditions, and as I thought adding the normalization step gave me the required. I tried using the built-in cv2.normalize(div,div,0,255,cv2.NORM_MINMAX) to normalize but it did not give me the required accuracy. So I had to come up with the normalization step myself after going through some examples in the internet.

To account for the fact that some images had poor quality, I tried adding a dropout layer but that did not improve accuracy too much so I gave up on that.
I was considering augmenting the data because of the un-uniformness of the distribution,
but I was already getting the required accuracy level without it so decided to go without that because I am already way behind the deadline.


My final model results were:
* training set accuracy of 100%
* validation set accuracy of 94%
* test set accuracy of 92%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image13] ![alt text][image14] ![alt text][image15] 
![alt text][image16] ![alt text][image17] ![alt text][image18]

The first image (Speed limit (60km/h) might be difficult to classify because there are other images that are similar and change among each other only by the numbers in them.
The other images seem to have distint characteristics so should be easier to predict, provided that enough test images of good quality are available.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					        |     Prediction	        				| 
|:-----------------------------:|:-----------------------------------------:| 
| Speed limit (60km/h)			| Speed limit (50km/h)   					| 
| Dangerous curve to the left	| Dangerous curve to the left				|
| Slippery road					| Slippery road								|
| Road Work	      				| Road Work					 				|
| Keep Left						| Keep Left      							|
| Keep Right					| Keep Right      							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. This compares favorably to the accuracy on the test set of 91%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Road work sign (probability of 0.54), and the image does contain a Road work sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .54         			| Road Work   									| 
| .49     				| Keep Right 									|
| .47					| Keep Left										|
| .56	      			| Slippery Road					 				|
| .46				    | Speed limit (50km/h) 							|
| .57				    | Dangerous curve to the left					|


The image with the lowest probability (0.46) was the one that the model got wrong. 
As expected it was the Speed limit (60km/h) image, when the model predicted it to be a  Speed limit (50km/h) image. Considering that 50 is so similar to 60, it is a pretty good guess though.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


