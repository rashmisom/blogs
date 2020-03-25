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

### We will use the official tokenization script created by the Google team
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py


 # Load BERT from the Tensorflow Hub
<pre><code><b>
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    bert_layer = hub.KerasLayer(module_url, trainable=True)
</b></code></pre>


# Next, we tokenize the data using the tf-hub model, which simplifies preprocessing:
<pre><code><b>
    # Load tokenizer from the bert layer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)  
  </b></code></pre>
We next build a custom layer using Keras, integrating BERT from tf-hub.

# Encode the text into tokens, masks and segments
<pre><code><b>
    train_input = bert_encode(train_df.clean_text.values, tokenizer, max_len=160)
    train_labels = train_df.target.values
  </b></code></pre>
  
# Build the model and train it
<pre><code><b>
    model_tweet_BERT = build_model(bert_layer, max_len=160)
    checkpoint = ModelCheckpoint('model_tweet.h5', monitor='val_loss', save_best_only=True)    
    bert_history = model_tweet_BERT.fit(
    train_input, train_labels,
    validation_split = 0.2,
    epochs = 4, 
    callbacks=[checkpoint],
    batch_size = 32,
    verbose=2
    )
  </b></code></pre>
  
## The build model
### This method will build the Model to be trained. This will take the output of the BERT later, send it to the sigmoid activation layer for classification.
 <pre><code><b>
def build_model(bert_layer, max_len=512):
    input_word_ids =  tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask =  tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids =  tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    x =  tf.keras.layers.Dropout(0.2)(clf_output)
    out =  tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model =  tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
    </b></code></pre>  
 
 ## The method will encode the 'text' column of train data. The BERT layer needs token, mask and the segment separator.
 <pre><code><b>
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []    
    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len      
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
    </b></code></pre>
     
# Test data and predict method
<pre><code><b>
    # Encode the text into tokens, masks and segments
    test_input = bert_encode(test_df.clean_text.values, tokenizer, max_len=160)
 

    # Build the model
    model_tweet_BERT = build_model(bert_layer, max_len=160)
    
    # load weights
    model_tweet_BERT.load_weights("model_tweet.h5")   

    y_pred = model_tweet_BERT.predict(test_input)
  </b></code></pre>



 