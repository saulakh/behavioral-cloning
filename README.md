## Behavioral Cloning
________________________________________
The goals / steps of this project are the following:
*	Use the simulator to collect data of good driving behavior
*	Build, a convolution neural network in Keras that predicts steering angles from images
*	Train and validate the model with a training and validation set
*	Test that the model successfully drives around track one without leaving the road
*	Summarize the results with a written report
________________________________________
### Files Submitted & Code Quality
##### Submission includes all required files and can be used to run the simulator in autonomous mode
My project includes the following files:
*	model.py containing the script to create and train the model
*	drive.py for driving the car in autonomous mode
*	model.h5 containing a trained convolution neural network
*	readme.md summarizing the results
##### Submission includes functional code
Using the Udacity provided [simulator](https://github.com/udacity/self-driving-car-sim) and my drive.py file, the car can be driven autonomously around the track by executing

python drive.py model.h5
##### Submission code is usable and readable
The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy
##### An appropriate model architecture has been employed
* My model consists of a convolutional neural network with 5 convolutional layers and 4 fully connected layers (model.py lines 81-96) based on the NVIDIA architecture model, and modified with additional dropout layers.
* The data is normalized in the model using a Keras lambda layer (model.py line 81), and the model includes RELU activation functions to introduce nonlinearity (model.py lines 83-87).
##### Attempts to reduce overfitting in the model
To reduce overfitting, I modified the model by adding dropout layers after the fully connected layers. I increased the dropout value to 0.3 to randomly remove more neurons. It was difficult to find a balance between reducing overfitting, while still achieving a successful run on the simulation track. The model typically drove worse as the validation loss decreased, and I felt confused about why this happened.

![image](https://user-images.githubusercontent.com/74683142/122601977-17295080-d040-11eb-9c1d-5bcfad89f763.png)
 
##### Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 99). The parameters I modified from the NVIDIA architecture were the dropout layers.
##### Appropriate training data
* The training data provided by Udacity was enough to keep the vehicle driving on the road. I recorded my own additional training data, but it was not necessary for completing the lap. I recorded recovery data for the turn after the bridge, which was where the model was initially driving off the track in simulation. 
* Overall, the driving seemed to get worse by including my own data. As a result, my recorded training data was not used in the final training model or saved in the project notebook.

### Model Architecture and Training Strategy
##### Solution Design Approach
* The overall strategy for deriving a model architecture was to start with a known architecture model (NVIDIA), see how well the vehicle could drive around the track, and modify the model accordingly.
* In order to gauge how well the model was working, I split 20% of my image and steering angle training dataset into a validation set. I started off using Udacity center images, and the vehicle drove well through the track until it reached a section with no barrier or lane markings between the road and dirt.
* I included images and measurements from the left and right cameras for more data, which made the driving jittery. I used a 0.4 correction to the steering angle to keep the vehicle centered in the lane. Next, I tried augmenting the data by flipping the images and steering angles from the center camera.

###### _Loss history before and after adding flipped center camera images:_
![image](https://user-images.githubusercontent.com/74683142/122603229-0aa5f780-d042-11eb-83ce-d68e509c101f.png) ![image](https://user-images.githubusercontent.com/74683142/122603279-15f92300-d042-11eb-8e18-a5393c301db3.png)

* I also tried different activation functions. Switching to an elu activation function improved the loss history plot, but in simulation the vehicle drove into the lake. So a lower validation loss did not seem to directly correlate to improved autonomous driving.

###### _Loss history with relu activation on the left, compared to elu on the right:_
![image](https://user-images.githubusercontent.com/74683142/122603599-9750b580-d042-11eb-98e9-bfead016fb42.png) ![image](https://user-images.githubusercontent.com/74683142/122603612-9d469680-d042-11eb-9f67-23bde032203e.png)

* Finally, I tried preprocessing the images differently. I used an image of the section where the vehicle was initially drifting off the track. Changing the color space helped to better differentiate between the dirt and road when there was no barrier or lane marking.
* After changing the color space to RGB, the vehicle was able to drive autonomously around the track without leaving the road. I also ran this model several times to see if it could complete laps consistently, and it successfully drove around the track each time.
##### Final Model Architecture
The final model architecture (model.py lines 80-96) was based on the NVIDIA architecture model.
| Layer |	Description |
|-------|---------------|
| Input/Normalization |	160x320x3 RGB image |
| Cropping 2D |	70x25 |
| Convolution 2D |	24 filters, 5x5 kernel, 2x2 stride, relu |
| Convolution 2D |	36 filters, 5x5 kernel, 2x2 stride, relu |
| Convolution 2D |	48 filters, 5x5 kernel, 2x2 stride, relu |
| Convolution 2D |	64 filters, 3x3 kernel, relu activation |
Convolution 2D | 64 filters, 3x3 kernel, relu activation |
| Dropout	| 0.3 |
| Flatten | |
| Dense | 100 |
| Dropout	| 0.3 |
| Dense | 50 |
| Dropout | 0.3 |
| Dense | 10 |
| Dropout | 0.3 |
| Dense | 1 |

##### Creation of the Training Set & Training Process
To capture good driving behavior, I relied on the data provided by Udacity. I was not great at keeping the car centered in the lane, so the vehicle did not drive as smoothly in my own training set. Since this could lower the accuracy of the steering angles from the training model, I tried not to include my own training data unless it was necessary. Here is an example image of center lane driving from the Udacity dataset:

![image](https://user-images.githubusercontent.com/74683142/122602329-a2a2e180-d040-11eb-8090-d6699042dc01.png)


I recorded the vehicle recovering from the sides of the road, so the vehicle would learn to steer back towards the center if it was close to the edges of the lane. These images show what a recovery looks like:

![image](https://user-images.githubusercontent.com/74683142/122602340-a8002c00-d040-11eb-94f1-6339ba1cf736.png) ![image](https://user-images.githubusercontent.com/74683142/122602355-adf60d00-d040-11eb-98ec-118414b7708b.png)
    
To augment the data set, I also flipped images and angles thinking that this would improve the model by expanding the amount of data to learn from. For example, here is an image that has then been flipped:

![image](https://user-images.githubusercontent.com/74683142/122602370-b5b5b180-d040-11eb-8f62-5ae80fa94ab0.png) ![image](https://user-images.githubusercontent.com/74683142/122602385-bb12fc00-d040-11eb-9c33-e6d23d311d4d.png)
   
To better differentiate between the road and the dirt, I tried changing the color space to HSV. In the end, I used the BGR2RGB function since it seemed to improve the consistency of simulation driving. Here is an image after applying the BGR2HSV function on the left, compared to the BGR2RGB function on the right:

![image](https://user-images.githubusercontent.com/74683142/122602407-c2d2a080-d040-11eb-8fa8-bb8818baf5f1.png) ![image](https://user-images.githubusercontent.com/74683142/122602420-c8c88180-d040-11eb-8668-43a1a4be1ca2.png)

After the data collection, I had 32,144 data points using images from all 3 camera angles, plus flipping the center camera images. The data was normalized in the model using a Keras lambda layer (model.py line 81). Finally, I randomly shuffled the data set, used 20% of the data for the validation set, and used the track run as the test set. 

The validation set helped determine if the model was over or under fitting, but optimizing the validation set did not seem to help as much as I expected. Instead, I relied on autonomous driving in simulation to design the model architecture. The vehicle was able to drive autonomously around the track without leaving the road, but I might continue to improve this project at a later time. I would like to smooth out steering angles by averaging recent measurements, since the driving is still a little jittery. I could also filter out some zero angle measurements so the vehicle is less biased towards driving straight, and hopefully stay centered within the lane better.

[Video Link](https://www.youtube.com/watch?v=imgtWWomFR4)
