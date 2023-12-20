### Setting up the Environment
To setup the virtual environment used in this repository run the following code:
```BASH
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
# Smart Start
In this repository we develop a neural network to detect the wakeup event from accelerometer data. Summarizing slides can be found [here](https://github.com/SonkyD/Smart-Start/blob/main/smart_start.pdf).

### The Problem
Getting out of bed in the morning can be a struggle—we're faced with the jarring cold light, a chilly bathroom, and the anticipation of waiting for the coffee machine to preheat. How can we improve this morning routine? The idea is to have smart systems recognizing our (soon) wakeup, gradually increasing light, warming up the bathroom, and having the coffee machine ready for a quick brew.

### Our Vision
Our aim is to leverage sensor data and machine learning to develop a model that recognizes when a person is waking up only by smart watch accelerometer data. This will trigger the appropriate setup of you home through a Smart Start/Smart Home app enabling a perfect start of the day!

### The Data
In our pursuit of relevant data, we found [this Kaggle Challenge]((https://www.kaggle.com/competitions/child-mind-institute-detect-sleep-states/data)), which provided wrist-worn accelerometer data comparable to that from ordinary smartwatches—widely worn during sleep. Prioritizing personal data protection, we opted for a minimal dataset. Given the intended app deployment on smartphones which limited internal storage and processing space, we chose to work with only accelerometer data.

### Data Analyis
During [exploratory data analysis (EDA)](https://github.com/SonkyD/Smart-Start/blob/main/EDA/EDA_clean.ipynb), we identified missing information on various events in the original event dataset and subsequently removed these instances. We noted that, the main difference between sleep and wakefulness is represented by more frequent changes in the  'ENMO' and 'z-angle'. </p> During feature engineering ([executed during preprocessing in this script]()) we focused on accentuating these differences. </p> 
![enmo and anglez over one night](/images/5AB450B3-D46A-4C57-AA7B-FF093AD3B268.jpeg)

### Model Development
Beginning with basic 'ENMO' and 'z-angle' measurements, a first [decision tree model](https://github.com/HPweck/sleepy_kid_Zzzz/blob/models/model_DecisionTree1.ipynb) demonstrated an initial accuracy of 88%. Incorporating newly engineered features significantly resulted in a [model](https://github.com/HPweck/sleepy_kid_Zzzz/blob/models/model_DecisionTree1_allData.ipynb) with an improved performance to 97.5%. 
![sleep vs wake detection](images/A2FDAA4A-8E44-438E-A817-21802AB7A99B_4_5005_c.jpeg)
In [error analysis](https://github.com/HPweck/sleepy_kid_Zzzz/blob/models/model_DecisionTree_allData_ErrorAnalysis.ipynb), we noticed proficient pattern recognition, with a slight bias towards sleep classification. 
Notably, misclassifications occurred predominantly during moments waking up. 
![misclassificated events](/images/2AE27D67-82BD-4011-BFE7-0162A7DD702D_4_5005_c.jpeg)
Because this is the most important event in terms of our business question, we further explored neural networks for classification.

### Final Model and Error Analysis
Our final 2 layer classical artificial neural network [ANN model]() predicts sleep and awake states with an accuracy of 97.68%.
It is better in prediction 'sleep' than 'awake' and is weak only at predicting the current state precisely during sleep onset and wakeup events:

### Conclusion
We could further imrpove the model by additional feature engineering such as calculated standard deviation, mean and maximum values over a longer period of time. For the usecase we could also include more sensor data like heart beat rate, body temperature and / or skin conductance, that can also easily be measured with common smart watches. In addition, we could refine the model by adding a personal 'wakeup' and /or sleep time window for the specific user. Besides that it would aso be possible to switch the focis a bit from prediciton of sleep vs. awake to prediciting the exact onset and wakeup event themselfs.

### Summary
Overall, we developed a machine learning model, more precisely an artificial neural network, that predicts sleep and awake state with an accuracy of 97.68 % only by using acceleration data. Our model outperfomrs a previously published Random Forest model which showed an accuracy of 74%. We achieved this by analyzing only acceleration data. Relying only on few, basic data is beneficial for the end user due to data privacy reasons. In summary, we developed a model that can be integrated into a smart home application triggering e.g preheating your coffee machine and warming up your bath and living room nabling you a great start into the day! 

| Contributors |
| --- |
| David Marks |
| Tabea Nöthe-Menchen |
| Hannes Peter Wecker |

### :two_hearts: Acknowledgement :two_hearts:
We are thankful to the Child Mind Institute for providing the data. We acknowledge Tim Schlange for his contributions during the start phase of this project. In addition, we thank all the neue fische coaches for introducing us to the world of data science and answerign our questions.

