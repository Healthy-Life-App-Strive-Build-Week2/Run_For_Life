# GoogleFit Journey Classifier

## A product by Team GoogleFit

![alt text](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdLGiada5O-NcK23jD9MsHJJ9uzmJ9C5qoKg&usqp=CAU "Logo Title Text 1")




*This program based on using AI to make life simpler.*

### THE TARGET

The purpose of this project was to produce a classification model that detects a journey type from from phone sensor readings.
    
The categories of journey we are trying to classify are;

1. On a train
2. On the road (e.g. bus/car)
3. Walking
4. Sitting still

![alt text](https://lh3.googleusercontent.com/proxy/MaaNQto0VBhOXQWmItpPckeVP1EjgNMn0hQmWRGAtZSFQWxGE20o8U9ummO7hoOHLgLJ_Csy_oNec0XE_Dqt9xxnUkZ1eVF1rIxxKs16-WGp0yy0DKJsAVpXvjy6d6VCBBjAh3Xq "Logo Title Text 1")

### THE DATA

We used a set of 229151 sensor readings taken from 100's of different journeys to train our model. The data set we used can be seen here.

![alt text](https://www.researchgate.net/profile/Charith-Perera-2/publication/234017923/figure/fig2/AS:667614586101765@1536183129256/Sensors-in-Mobile-Phones.png "Logo Title Text 1")


### WHAT WE DID WITH THE DATA

We first ran some functions on the data to decide where a recording started and ended, this would prove vital
later for getting consistencies over time windows as the dataset was organised in such a way that time had lost
it's true meaning.

In later production when we take live data from a device this step will not be necessary. We did it at this stage of development to replicate sequential sensor readings as the original data sometimes is not correctly ordered.

![alt text](https://files.realpython.com/media/How-to-Plot-With-Pandas_Watermarked.f283f64b4ae3.jpg "Logo Title Text 1")


### HALT, DATA LEAKS!

Our first models ran at 97-99% accuracy with no tuning, we were incredibly suspicous of our results. We isolated a new user that the model hasn't seen before we found that 'user traits' were leaking test data to our training model.

We then took the approach to isolate individual users for testing, our predictions dropped to 45% accuracy, now we had some work to do!

After taking sometime to analyze the feature importance and different combination of ML models we were able to improve this upto a whopping 85.14% over 4 classes and 97% over 3 classes.

![alt text](https://www.zivver.eu/hubfs/Data_Breach_vs.%20Data_leak_explained_zivve_blog_en.jpg "Logo Title Text 1")
