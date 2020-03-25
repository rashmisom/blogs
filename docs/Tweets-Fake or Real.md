---
layout: default
title: Real or Not? NLP with Disaster Tweets
description: Real or Not? NLP with Disaster Tweets
repository_url: https://github.com/rashmisom/Tweets-NLP-sentiment
---

Twitter has become an important communication channel in times of emergency. The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time. Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).
In this competition, you’re challenged to build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

In this blog, let us discuss one approach to solve this problem statement.<br>
 

---

 
## Problem Statement

To build a machine learning model that predicts which Tweets are about real disasters and which one’s aren’t.

---

### Data Format

We have access to a dataset of 10,000 tweets that were hand classified.
We are predicting whether a given tweet is about a real disaster or not.<br>
If so, predict a 1<br>
If not, predict a 0

Click [here](https://www.kaggle.com/c/nlp-getting-started){:target="_blank"} for more details.

---

### The features involved are

1. id - a unique identifier for each tweet 
2. text - the text of the tweet 
3. location - the location the tweet was sent from (may be blank) 
4. keyword - a particular keyword from the tweet (may be blank) 
5. target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

---

## Mapping the Business problem to a Machine Learning Problem 


1. Prepare data for the model¶
### Type of Machine Learning Problem

<b><i>It is a binary classification problem.</i></b>

 # We will use the official tokenization script created by the Google team
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py


Next, we tokenize the data using the tf-hub model, which simplifies preprocessing:
<pre><code><b>
    # Load tokenizer from the bert layer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)  
  </b></code></pre>
We next build a custom layer using Keras, integrating BERT from tf-hub.

### Helper Functions



 
