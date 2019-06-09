# Load libraries
import sys
# scipy
import scipy
# numpy
import numpy
# matplotlib
import matplotlib
# pandas
import pandas
# scikit-learn
import sklearn
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import os
# Check all library version
print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))
# Display all of the files found in your current working directory
print(os.getcwd())
print(os.listdir(os.getcwd()))
# Load twitter dataset - we will check sentiment later 
url = "data/new-year-resolution-twitter-2015.csv"
dataset = pandas.read_csv(url, encoding = "ISO-8859-1")
# Check dataset shape
print ("#########SHAPE######################")
print(dataset.shape)
print ("#########DESCRIPTION################")
# Check dataset description
print(dataset.describe())
# Check class distribution by gender
print ("#########GENDER#####################")
print(dataset.groupby('gender').size())
# Check class distribution by tweet_region
print ("#########REGION######################")
print(dataset.groupby('tweet_region').size())

dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

# scatter plot matrix
scatter_matrix(dataset)
plt.show()
