#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt


# In[2]:


#Path
loc = 'data\\'

# Read data
df_train = pd.read_csv(loc + '17-18-pca-std.csv')
df_test = pd.read_csv(loc + '2019-pca-std.csv')


train_x = df_train.iloc[:,1:8]
train_y = df_train.iloc[:,25]

test_x = df_test.iloc[:,1:8]
test_y = df_test.iloc[:,25]

print(train_x)
print(train_y)
print(test_x)
print(test_y)


# In[3]:


# Create model with optimal kernel and parameters
model = SVC(kernel='sigmoid', C=20, gamma=.1, coef0=.2)
# Fit model and test model
model.fit(train_x, train_y)
print(model.score(test_x, test_y))


# In[4]:


clf = svm.SVC(decision_function_shape='ovo')
clf.fit(test_x.values, test_y.values)

value=1.5
width=0.75

# Plot Decision Region using mlxtend's awesome plotting function
plot_decision_regions(train_x.values, train_y.values, clf=clf, feature_index=[0, 1], 
                      filler_feature_values={2: value, 3:value, 4:value, 5:value, 6:value},
                      filler_feature_ranges={2: width, 3: width, 4:width, 5:width, 6:width})



# Update plot object with X/Y axis labels and Figure Title
plt.xlabel('Components 1', size=14)
plt.ylabel('Components 2', size=14)
plt.title('SVM Decision Region Boundary', size=16)
plt.savefig(loc + 'decisionboundary.png')


# In[7]:


# Plot decision function for all training points
plt.plot(model.decision_function(train_x))
plt.style.use('ggplot')
plt.savefig(loc + 'decisionfunction.png')
plt.show()


# In[8]:


# Plot support vectors
print(len(model.support_))
plt.plot(model.support_vectors_)
plt.style.use('ggplot')
plt.savefig(loc + 'supportvectors.png')
plt.show()


# In[ ]:




