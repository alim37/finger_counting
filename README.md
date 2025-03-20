# finger counting model with live webcam

<p align="center">
  <img src="https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmVqazJhNTc4czBjd2duNWwza3R5Y2JnMmFhOW1jc3JwZHNvenBxdyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/Nz9ev3wZDaSMoqdgwX/giphy.gif" alt = "1-5 Finger Counting Showcase"/>
  <br>
  <em>1-5 Finger Counting Showcase</em>
</p>

## DATA
Since I wanted to implement this trained CNN on live webcam, I wanted a larger data set. Found a [github repo](https://github.com/Paradiddle131/Finger-Count-Detection-Using-Deep-Learning?tab=readme-ov-file) which had [videos](https://drive.google.com/drive/folders/143LEc5sai_ReSzNSxKrkXKxH5iZl_XL4) that could be processed into a larger set of training & testing data.

## MAIN
The function of `main.py` is to process the data, split into training and testing sets, and then build/train the model. The processing of the data is done through MediaPipe even though we don't need to to maintain consistency in data representation and to avoid runtime conflicts when testing with the live webcam.
## WEBCAM TEST
Live webcam test will load the model trained from `main.py` and use it to make a prediction off of the live webcam. 

## resources
- compiled list of resources used:
  - MediaPipe python image processing:
    - https://mediapipe.readthedocs.io/en/latest/solutions/hands.html
  - TensorFlow:
    - https://www.tensorflow.org/api_docs/python/tf/keras/Model
  - SciKit (train_test_split):
    - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    - https://stackoverflow.com/questions/42191717/scikit-learn-random-state-in-splitting-dataset
