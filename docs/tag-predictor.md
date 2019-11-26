---
layout: default
title: Stack Overflow Tag Prediction
description: Lets suggest the tags
repository_url: https://github.com/rashmisom/tag-prediction
---

We all know [stackoverflow](https://stackoverflow.com){:target="_blank"}. It has questions and answers. Now, do we know these questions are tagged? A tag is a word or phrase that describes the topic of the question. Tags are a means of connecting experts with questions they will be able to answer by sorting questions into specific, well-defined categories. Tags can also be used to help you identify questions that are interesting or relevant to you.
  
In this blog, let us discuss the tagging predictor.<br>
<b>stackoverflow tag predictor</b> we have analysed this ML problem as part of [appliedaicourse](https://www.appliedaicourse.com/){:target="_blank"}.
So lets get started on understanding the complete but simple steps to do the TAG PREDICTION.

---

 
## Problem Statement

Predict the tags (a.k.a. keywords, topics, summaries), given only the question text and its title.

Read the full problem statement on [kaggle](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/){:target="_blank"} 

## Business Objectives and Constraints

1. Predict tags with high [precision and recall](https://goo.gl/csnXGo){:target="_blank"}
3. No strict latency constraints.

---

### Data Format

Data contains 4 fields

1. Id - Unique identifier for each question

2. Title - The question's title

3. Body - The body of the question, contains text description and may also contain code snippet

4. Tags - The tags associated with the question

Click [here](https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data){:target="_blank"} for more details.

### Sample data point

<b>Id</b>: 5

<b>Title:</b>Implementing Boundary Value Analysis of Software Testing in a C++ program?

<b>Body:</b>

<pre><code>
        #include&lt;
        iostream&gt;\n
        #include&lt;
        stdlib.h&gt;\n\n
        using namespace std;\n\n
        int main()\n
        {\n
                 int n,a[n],x,c,u[n],m[n],e[n][4];\n         
                 cout&lt;&lt;"Enter the number of variables";\n         cin&gt;&gt;n;\n\n         
                 cout&lt;&lt;"Enter the Lower, and Upper Limits of the variables";\n         
                 for(int y=1; y&lt;n+1; y++)\n         
                 {\n                 
                    cin&gt;&gt;m[y];\n                 
                    cin&gt;&gt;u[y];\n         
                 }\n         
                 for(x=1; x&lt;n+1; x++)\n         
                 {\n                 
                    a[x] = (m[x] + u[x])/2;\n         
                 }\n         
                 c=(n*4)-4;\n         
                 for(int a1=1; a1&lt;n+1; a1++)\n         
                 {\n\n             
                    e[a1][0] = m[a1];\n             
                    e[a1][1] = m[a1]+1;\n             
                    e[a1][2] = u[a1]-1;\n             
                    e[a1][3] = u[a1];\n         
                 }\n         
                 for(int i=1; i&lt;n+1; i++)\n         
                 {\n            
                    for(int l=1; l&lt;=i; l++)\n            
                    {\n                 
                        if(l!=1)\n                 
                        {\n                    
                            cout&lt;&lt;a[l]&lt;&lt;"\\t";\n                 
                        }\n            
                    }\n            
                    for(int j=0; j&lt;4; j++)\n            
                    {\n                
                        cout&lt;&lt;e[i][j];\n                
                        for(int k=0; k&lt;n-(i+1); k++)\n                
                        {\n                    
                            cout&lt;&lt;a[k]&lt;&lt;"\\t";\n               
                        }\n                
                        cout&lt;&lt;"\\n";\n            
                    }\n        
                 }    \n\n        
                 system("PAUSE");\n        
                 return 0;    \n
        }\n
        </code></pre>

<b>Tags</b>:'c++ c'

---
## Mapping the Business problem to a Machine Learning Problem 

### Type of Machine Learning Problem

<b><i>It is a multi-label classification problem.</i></b>

In Multi-label Classification, multiple labels (in this problem its tags) may be assigned to each instance and there is no constraint on how many of the classes the instance can be assigned to.
Source: [Wiki](https://en.wikipedia.org/wiki/Multi-label_classification){:target="blank"}

Find more about multi-label classification problem [here](http://scikit-learn.org/stable/modules/multiclass.html){:target="blank"}

A question on Stackoverflow might be about any of C, Pointers, JAVA, Regex, FileIO and/or memory-management at the same time or none of these.

### Performance metric

<b>Micro-Averaged F1-Score (Mean F Score) </b>: 
The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

<i>F1 = 2 * (precision * recall) / (precision + recall)</i><br>

In the multi-class and multi-label case, this is the weighted average of the F1 score of each class. <br>

<b>'Micro f1 score': </b><br>
Calculate metrics globally by counting the total true positives, false negatives and false positives. This is a better metric when we have class imbalance.
<br>

<b>'Macro f1 score': </b><br>
Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
<br>

https://www.kaggle.com/wiki/MeanFScore <br>
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html <br>
<br>
<b> Hamming loss </b>: The Hamming loss is the fraction of labels that are incorrectly predicted. <br>
https://www.kaggle.com/wiki/HammingLoss <br>

---
## EDA (Exploratory Data Analysis)

I have used [pandas](https://pandas.pydata.org/){:target="blank"} library to load the data. Please visit [my github repo](https://github.com/SachinKalsi/machine-learning-case-studies/tree/master/stackoverflow_tag_preditor){:target="_blank"} to see the full code. I have taken a sample of 1000000 (10 lakh) data points from Train.csv. Here is a list of major observations from EDA.

1. <b>Number of rows in the database:</b> 1000000
2. <b>5.6% of the questions are duplicate:</b> Number of rows after removing duplicates:  943582
3. <b>Number of unique tags:</b> 34945
4. <b>Top 10 important tags:</b>  ['.a', '.app', '.aspxauth', '.bash-profile', '.class-file', '.cs-file', '.doc', '.drv', '.ds-store', '.each']

5. Few number of tags have appeared more than 50000 times & the top 25 tags have appeared more than 10000 times
![Distribution of number of times tag appeared in questions(for first 100 tags)]({{site.baseurl}}data/images/stackoverflow/tag_counts.png)
6. <b>Tags analysis</b>
    1. Maximum number of tags per question: 5
    2. Minimum number of tags per question: 1
    3. Avg. number of tags per question: 2.887779
    4. Questions with 3 tags appeared more in number
    ![Number of tags in the question]({{site.baseurl}}data/images/stackoverflow/question_with_tag_frequency.png)
    5. Word cloud of tags
    ![Word cloud of tags]({{site.baseurl}}data/images/stackoverflow/word_cloud.png)
    6.`C#` appears most number of times, `Java` is the second most. Majority of the most frequent tags are programming language. And here is the chart for top 20 tags
    ![frequency of top 20 tags]({{site.baseurl}}data/images/stackoverflow/frequency_of_top_20_tags.png)

---    
    
## Cleaning and preprocessing of Questions

<i>P.S: <b>Due to hardware limitations, I am considering only 500K data points</b></i>

### preprocessing
<ol>
  <li>56.37% percentage of questions contains HTML tag &lt;code&gt; tag. So separate out code-snippets from  the Body</li>
  <li>Remove Spcial characters from title and Body (not in code)</li>
  <li><b>Remove stop words (Except 'C')</b></li>
  <li>Remove HTML Tags</li>
  <li>Convert all the characters into small letters</li>
  <li>Use SnowballStemmer to stem the words.<br><br>
  <i>Stemming is the process of reducing a word to its word stem. <br>
  <b>For Example:</b> "python" is the stem word for the words ["python" "pythoner", "pythoning","pythoned"]</i></li>
  <li><b>Give more weightage to title: Add title three times to the question</b>. Title contains the information which is more specific to the question and also only after seeing the question title, a user decides whether to look into the question in detail. At least most of the users do this if not all </li>
</ol>

<h5>Sample question after preprocessing:</h5>

>"modifi whoi contact detail modifi whoi contact detail modifi whoi contact detail use modifi function display warn mesag pleas help modifi contact detail"

---
## Machine Learning Models

<i>
Total number of questions: 500000<br>
Total number of tags: 30645
</i>

Here we are going to use <i><b>Problem Transformation(Binary Relevance)</b></i> method to solve the problem.

<h4>Binary Relevance:</h4> Here we are going to convert multi-label classification problem into multiple single class classification problems.For example if we are having 5 multi-label classification problem, then we need to train 5 single class classification models.

 Basically in this method, we treat each label (in our case its tag) as a separate single class classification problem. This technique is simple and is widely used.

Please refer to [analytics vidhya's blog](https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/){:target="_blank"} to know more about the techniques to solve a Multi-Label classification problem.

<h4>Downscaling of data</h4>
Coming back to our stackoverflow predictor problem, we need to train 30645 models literally!!!
Thats really huge (both in terms of time & speed) for a system with 8GB RAM & i5 processor. So we will sample the number of tags instead considering all of them. But how many tags to be sampled with the minimal information loss ? Plotting 'percentage of questions covered' Vs 'Number of tags' would help to solve this.

!['percentage of questions covered' Vs 'Number of tags']({{site.baseurl}}data/images/stackoverflow/percentage_of_questions_covered.png)

<i>Observations</i>
<ol>
  <li>with  500 tags we are covering  89.782 % of questions</li>
  <li>with  600 tags we are covering  91.006 % of questions</li>
  <li>with  5500 tags we are covering  99.053 % of questions</li>
</ol>

By choosing only 600 tags (2% approximately) of the total 30645 tags we are loosing only 9% of the questions & also training 600 models is reasonable (Of course it also depends on the type of machine learning algo we choose). So we shall choose 600 tags.

<h4>Train and Test data</h4>

If the data had timestamp attached for each of the questions, then splitting data with respect to its temporal nature would have made more sense than splitting data randomly. But since the data is not of temporal nature (i.e., no timestamp), we are splitting data randomly into 80% train set & 20% test set

<pre><code><b>train_datasize= 0.8 * preprocessed_title_more_weight_df.shape[0]
x_train = preprocessed_title_more_weight_df[:int(train_datasize)]
x_test = preprocessed_title_more_weight_df[int(train_datasize):]
y_train = multilabel_yx[0:train_datasize,:]
y_test = multilabel_yx[train_datasize:,:]
</b></code></pre>

<h4>Featurizing Text Data with TfIdf vectorizer</h4>

There are various ways to featurize text data. I have explained this deeply in my [blog](https://goo.gl/g1cB6z){:target="_blank"} post. First lets featurize the question data with TfIdf vectorizer. [TfidfVectorizer](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html){:target="_blank"} of sklearn helps here

<pre><code><b>vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2",sublinear_tf=False, ngram_range=(1,3))
x_train_multilabel = vectorizer.fit_transform(x_train['questions'])
x_test_multilabel = vectorizer.transform(x_test['questions'])
</b></code></pre>

Dimensions of train data X: (400000, 90809) Y : (400000, 600)

Dimensions of test data X: (100000, 90809) Y: (100000, 600)

<h5>Applying Logistic Regression with OneVsRest Classifier (for tfidf vectorizers)</h5>

Lets use Logistic Regression algo to train 600 models (600 tags). We shall use [OneVsRestClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html){:target="_blank"} of sklearn to achieve the same

<pre><code><b>classifier = OneVsRestClassifier(LogisticRegression(penalty='l1'), n_jobs=-1)
classifier.fit(x_train_multilabel, y_train)
predictions = classifier.predict(x_test_multilabel)
</b></code></pre>

<b><u>Results</u></b>

Micro F1-measure: 0.4950

Macro F1-measure: 0.3809

<h4>Featurizing Text Data with Bag Of Words (BOW) vectorizer</h4>

This time lets featurize the question data with BOW upto 4 grams.

<i><b> I did try Featurizing Text Data with Bag Of Words, but my system was giving out of memory error.</b> So again I have to downscale the data to 100K.</i> Here is train & test data after downscaling

Dimensions of train data X: (80000, 200000) Y : (80000, 600)

Dimensions of test data X: (20000, 200000) Y: (20000, 600)

<pre><code><b>vectorizer = CountVectorizer(min_df=0.00001,max_features=200000, ngram_range=(1,4))
x_train_multilabel = vectorizer.fit_transform(x_train['questions'])
x_test_multilabel = vectorizer.transform(x_test['questions'])</b></code></pre>

 
