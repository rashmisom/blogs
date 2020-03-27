---
layout: default
title: Face Forensic
description: Learning to Detect Manipulated Facial Images
repository_url: https://github.com/rashmisom/Face-Forensic
---

## Problem Statement


The rapid progress in synthetic image generation and manipulation has now come to a point where it raises significant concerns on the implication on the society. At best, this leads to a loss of trust in digital content, but it might even cause further harm by spreading false information and the creation of fake news. 
In this blog, we will build a machine learning model that learns to detect manipulated facial images. We have access to a dataset of fake and real videos. We have to train a model to predict whether the given video(or image sequence) is fake or real.
For more details, kindly go through this [link](https://www.groundai.com/project/faceforensics-learning-to-detect-manipulated-facial-images/1)

In this blog, let us discuss one approach to solve this problem statement.<br/>
 

---

 
## Our Approach
 
Our objective is to build a model which would recognize whether the given video is **Real** or **Fake**.
Let us summarize the steps to solve the problem at hand: 
1. We will download the dataset from the [FaceForensics](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md) 
2. For our dataset, we chose two computer graphics-based approaches  (**Face2Face** and **FaceSwap** ) and a deep-learning-based approach  (**Deepfakes** ). These will give us manipulated videos.  
3. We will also download the original youtube videos from there.  
4. We will use the following script to download all images or videos. <br/>
      `download-FaceForensics.py`
5. We will have to extract the sequence images from these videos. We will  use the following script provided by FaceForensics for extracting the  frames.<br/>  
      `python extracted_compressed_videos.py <output path> -d <"all" or single dataset via "Face2Face" or "original"> -c c0`
6. We will process all extracted images with a standard Dlib face detector to  get the face bounding boxes for further processing.  
7. We split the dataset into a fixed training, validation and test set.  
8. We build a CNN model to extract the image features and learn the weights to make the prediction of the manipulated videos.  
9. The above paper suggests that the classification based on XceptionNet outperforms all other variants in detecting fakes, we will also try to build our  model using XceptionNet .  
10. Once our CNN model is trained, we will predict the image sequences validity of the test videos.

---

## Data Collection

We will consider around 150 _fake_ and 151 _Original_ vedios for our model.
1. We will download 50 fake videos of each type.<br/>
    `python faceforensics_download_v4.py data\fake -d Deepfakes -c c23 -t videos -n 50`
    `python faceforensics_download_v4.py data\fake -d Face2Face -c c23 -t videos -n 50`
    `python faceforensics_download_v4.py data\fake -d FaceSwap -c c23 -t videos -n 50`

2. We will download 151 original videos.<br/>    
    `python faceforensics_download_v4.py data\original -d youtube -c c23 -t videos -n 151`

3. We will extract the image sequence from these videos using the following commands.<br/>
    `python extracted_compressed_videos.py --data_path data\fake -d Deepfakes -c c23`
    `python extracted_compressed_videos.py --data_path data\fake -d Face2Face -c c23`
    `python extracted_compressed_videos.py --data_path data\fake -d FaceSwap -c c23`
    `python extracted_compressed_videos.py --data_path data\original -d original -c c23`

4. We will go through these image sequences and collect 50 image sequence for EACH video and put all these in the folder       "data_set". These images will be further processed and split into <br/>
 train, cv and test data.
 

## The features involved are

1. id - a unique identifier for each tweet
2. text - the text of the tweet
3. location - the location the tweet was sent from (may be blank)
4. keyword - a particular keyword from the tweet (may be blank)
5. target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)

## EDA (Exploratory Data Analysis)

Let us analyse the data a bit.

 1. <b>Lets check on the 'target', the dependent variable distribution:</b>
      <pre><code><b>
        sns.barplot(target_value_count.index,target_value_count.values,palette="rainbow")
      </b></code></pre>
 ![Target Distribution](../images/target_distribution.png)
  
 2. <b> Checking for the TOP 'keywords'</b>
   <pre><code><b>
      keyword_value_count = train_df["keyword"].value_counts()
      plt.barh(y=list(keyword_value_count.index)[:10],width=keyword_value_count.values[:10],color= 'rgbkymc')
   </b></code></pre>
 ![Top Keywords](../images/top_keywords.png)
   
3. <b>Distribution of 'keywords' for Real and Fake tweets:</b> 
The complete code for <i>univariate_barplots</i> is available [here](https://github.com/rashmisom/Tweets-NLP-sentiment)
 <pre><code><b>
      univariate_barplots(train_df,'keyword','target',1,21) 
 </b></code></pre>
 ![Keyword Distribution](../images/keyword_distribution.png)
 
4. <b>Distribution of 'Location' for Real and Fake tweets:</b> 
<pre><code><b>
     univariate_barplots(train_df,'location','target',1,41)
</b></code></pre>
 ![Location Distribution](../images/location_distribution.png)
  
 5. <b>Lets see the 'Number of words" in the tweets:</b> 
<pre><code><b>
     disaster_word_count = train_df[train_df['target']==1]['text'].str.split().apply(len)
     disaster_word_count = disaster_word_count.values
     fake_word_count = train_df[train_df['target']==0]['text'].str.split().apply(len)
     fake_word_count = fake_word_count.values
     # the box plot
     plt.boxplot([disaster_word_count, fake_word_count])
     # the distribution plot
     sns.distplot(disaster_word_count, hist=False, label="Real Disaster")
     sns.distplot(fake_word_count, hist=False, label="Fake Disaster")

</b></code></pre>
 ![Number of words](../images/num_of_words.png)
 <br>
 ![Number of words](../images/num_words_dist.png)
 
 6. <b>Number of characters in the tweet text:</b> 
 <pre><code><b>
     ax1.hist(real_char_len,color='blue')
     ax2.hist(fake_char_len,color='green')
</b></code></pre>
 ![Number of Characters](../images/num_of_char.png)
 
  7. <b>Average word length in a tweet text:</b> 
 <pre><code><b>
     sns.distplot(real_disaster_word_count.map(lambda x: np.mean(x)),ax=ax1,color='blue')
     sns.distplot(fake_disaster_word_count.map(lambda x: np.mean(x)),ax=ax2,color='green')
</b></code></pre>
 ![Average word length](../images/avg_word_len.png)
 
   8. <b>The punctuation marks in the tweets:</b> 
 <pre><code><b>
     sns.distplot(real_disaster_punctuation_marks,ax=ax1,color='blue')
     sns.distplot(fake_disaster_punctuation_marks,ax=ax2,color='green')
</b></code></pre>
 ![Punctuation Marks](../images/punctuation_marks.png)
 
   9. <b>Word Cloud for the real and fake disaster tweets:</b> 
 <pre><code><b>
     tweet_wordcloud(train_df[train_df["target"]==1], title="Train Data Tweets of Real Disaster")
     tweet_wordcloud(train_df[train_df["target"]==0], title="Train Data Tweets of Fake Disaster")
</b></code></pre>
 ![Real tweets](../images/wc1.png) <br>
 ![Fake tweets](../images/wc2.png)
 
## Mapping the Business problem to a Machine Learning Problem

### Prepare data for the model
We load the data from the train.csv and test.csv files.
<pre><code><b>
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
  </b></code></pre>
 
 Once we have the data loaded, we must preprocess the data before submitting it to the ML model for training.
 Lets look into abstract of the data preprocessing and the details of the same is available [here](https://github.com/rashmisom/Tweets-NLP-sentiment) .

<pre><code><b>   
        ## decontract the text,remove html etc
        sent = cleanText(sentance)
        ## some more data updates
        sent = sent.replace('\\r', ' ').replace('\\"', ' ').replace('\\n', ' ').replace(",000,000", "m")\
                           .replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will")
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        ## correct the spellings
        sent = correct_spellings(sent)
        ## remove the stopwords
        sent = ' '.join(e for e in sent.split() if e not in stopwords and e not in punctuations)
  </b></code></pre>

 
## How to use BERT for text classification. Lets look into the steps one by one:
### We will use the official tokenization script created by the Google team.
<pre>
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py </pre>

The processes of tokenisation involves splitting the input text into list of tokens that are available in the vocabulary. <br>

### Let us load BERT from the Tensorflow Hub:
<pre><code><b>
    module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
    bert_layer = hub.KerasLayer(module_url, trainable=True)
</b></code></pre>

### Next, we prepare the tokenizer using the tf-hub model:
In order to pre-process the input and feed it to BERT model, we need to use a tokenizer.
<pre><code><b>
    # Load tokenizer from the bert layer
    vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
    do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
    tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)  
  </b></code></pre>
We next build a custom layer using Keras, integrating BERT from tf-hub.

### Encode the text into tokens, masks and segments:
BERT requires pre-processed inputs. It supports the tokens signature, which assumes pre-processed inputs: input_ids, input_mask, and segment_ids. To achieve this, we encode the data using the tokenizer built in the previous step. 
The deails of the method <u>bert_encode</u> will be discussed later.
<pre><code><b>
    train_input = bert_encode(train_df.clean_text.values, tokenizer, max_len=160)
    train_labels = train_df.target.values
  </b></code></pre>
  
### Build the model and train it:
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
  
### Lets check the layers of our model. We will build the model using KERAS layer on top of the BERT layer.
This method will build the Model to be trained. This will take the output of the BERT later, send it to the sigmoid activation layer for classification.
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
 
### Lets look into the bert_encode method which we are using to encode the text data for feeding it to the BERT layer.
 The method will encode the 'text' column of train data. The BERT layer needs token, mask and the segment separator.
 We first tokenize the sentence using the tokenizer created from vocab.txt . We add [CLS] to start of sentence and [SEP] to the end of the sentence. Finally, we pad the sentence with 0. As we are dealing with one sentence per example, we set segment_id to be 0 and further we set mask to 1 for all tokens.We set this mask to 0 beyond the number of tokens.
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
     
### Last but not the least, lets not forget the Test data and predict method:
<pre><code><b>
    # Encode the text into tokens, masks and segments
    test_input = bert_encode(test_df.clean_text.values, tokenizer, max_len=160)
    # Build the model
    model_tweet_BERT = build_model(bert_layer, max_len=160)    
    # load weights
    model_tweet_BERT.load_weights("model_tweet.h5")   
    y_pred = model_tweet_BERT.predict(test_input)
  </b></code></pre>

## Model performance:
The model is a binary classification model and we can check the accuracy of the trained model and plot the accuracy and loss graphs to check on the model performance.
<pre><code><b>
    print(bert_history.history["accuracy"])
    print(bert_history.history["val_accuracy"])    
    plot_graphs(bert_history,"accuracy")
 </b></code></pre>
 
<br/>How can we forget the utility method
<pre><code><b>
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()    
  </b></code></pre>
  
### Kaggle Submission
 
On submitted the predicted values for the test dataset, the kaggle score came as shown in the image below and this can be further    improved by the suggestions listed in the <i>Future Work</i> section of this blog.
![Kaggle Submission](../images/kaggle_sub2.png)
 
 
### Future work
  1. The performance of the model can be further improved by fine tuning the hyper parameters of the model.
  2. Instead of fine tuning the BERT module, we can try to train only the top two or top four layers and check out result.

## References
 
 1. I have done this case study as part of [appliedaicourse](https://www.appliedaicourse.com/)
 2. [BERT](https://github.com/google-research/bert)
 3. [Illustrated BERT](http://jalammar.github.io/illustrated-bert/)
 4. [HuggingFace](https://github.com/huggingface/transformers)
 