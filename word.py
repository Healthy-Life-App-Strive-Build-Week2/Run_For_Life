HELLO!!!


We are team google fit
Just a little bit about the team;

Our frontend developers were Umut & Daniel
And I, Mark was the backend developer.


We set out to utilise phone sensor readings to classify
what the phone user was doing while the sensor readings
were being recorded. We wanted to take these classifications
and build a useful application from them.

Our final idea was as follows;

The App will calculate approximately how many calories
 you're burning right now, based on your current activity.
 It will work by a classification of your activity from mobile phone
 sensor data input into a trained ML model. It gives you an overview
  of how many calories you burned during the day. It also
  tells you in textform how you spent your day for example:
   'You spent 2 hours on the train'
   'You have been sitting for 6 hours'
    'You walked today one hour'.

JUST A NOTE FROM OUT DEVELOPMENT TEAM:
Our multiple platform app is still under construction.
We will demonstrate the functionality through streamlit
today.

1. Creating the classifier
- We started out by doing some EDA and ran some machine
learning models to get an overview. This highlighted key
areas to address
- quantity of classifications
        Changing had a big impact on accuracy

- data leakage
        Original models were having 99% accuracy (OOPS!)

- runtimes on the model
        the best models took forever to train

- How we overcame them:
- Data leaks: we Created test users that the model had never seen,
- quantity of classifications
        : We experimented with different combinations of classes.
            (combined bus and car into road in the end)
            (we can actually can combine train into bus & road)
            defining a transport category) and this gets 97%
            but we felt a deeper range of classifications was better.

- Runetimes: The longest part is training, especially
            for our voting classifier (more on that soon)
            that has to train 4 models which takes
            approximately 7 mins.

            We created our own model class that we were able to 'pickle'.
            For those unaware it saves a snapshot of the trained models
            that we can export into our app.

            - Allowing the user to run a set of predctions that sub 1 second.


Go into live demonstrations

Go to how it works
