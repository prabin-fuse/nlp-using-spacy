# nltk-using-spacy

## Text Preprocessing Using Spacy

### Stop Words

- Stop words are words that are filtered out before or after the natural language data(text) are processed.
- stop words typically refers to the most common words in a language.
- There is no universal list of stop words that is used by all NLP tools in common.

**what are stop words?**
- Stopwords are the words in any language which does not add much meaning to a sentence.
- They can safely be ignored without sacrificing the meaning of the sentence.
- For some search engines, these are some of the most common, short function words, such as the, is, at, which, and on.

**But sometimes, stop words can be really useful and shouldnot be removed.**

**When to remove stop words?**

- If we have a task of text classification or sentiment analysis then we should remove stop words as they do not provide any information to our model i.e. keeping out unwanted words out of our corpus.
- But, if we have the task of language translation then stopwords are useful, as they have to be translated along with other words.
- There is no hard and fast rule on when to remove stop words
    1) Remove stopwords if task to be performed is one of Language Classification, Spam Filtering, Caption Generation, Auto-Tag Generation, Sentiment analysis, or something that is related to text classification.
    2) Better not to remove stopwords if task to be performed is one of Machine Translation, Question Answering problems, Text summarization, Language Modeling.

**Pros of Removing stop words**

- Stopwords are often removed from the text before training deep learning and machine learning models since stop words occur in abundance, hence providing little to no unique information that can be used for classification or clustering.
- On removing stopwords, dataset size decreases, and the time to train the model also decreases without a huge impact on the accuracy of the model.
- Stopword removal can potentially help in improving performance, as there are fewer and only significant tokens left. Thus, the classification accuracy could be improved.

**Cons of Removing Stop Words**

- Improper selection and removal of stop words can change the meaning of our text. So we have to be careful in choosing our stop words.
- Example: This movie is not good
    - If we remove (not ) in pre-processing step the sentence (this movie is good) indicates that it is positive which is wrongly interpreted.

**Removing Stop words using SpaCy Library**

- Comparing to NLTK, spacy got bigger set of stop words (326) than that of NLTK (179)
- installation: (spacy, English Language Model)
    - pip install -U spacy
    - python -m spacy download en_core_web_sm



## Tokenization:
- Tokenization refers to diving the whole text into multiple managable units.
- Helps to form sequence of words or sentences.
- Each tokens have meaning and semantic relation with other tokens.

- Word Tokenization
    - Word Tokenization simply means splitting sentence/text in words.
    - Using attribute `token.text` to tokenize the doc

- Sentence Tokenization

    - Sentence Tokenization is the process of splitting up strings into sentences.
    - A sentence usually ends with a full stop (.), here focus is to study the structure of sentence in the analysis
    - use `sents` attribute from spacy to identify the sentences.


## Punctuation:
- punctuation are special marks that are placed in a text to show the division between phrases and sentences.
- There are 14 punctuation marks that are commonly used in English grammar.
- They are, **period, question mark, exclamation point, comma, semicolon, colon, dash, hyphen, parentheses, brackets, braces, apostrophe, quotation marks, and ellipsis**.
- We can remove punctuation from text using `is_punct` attribute.


## Lower Casing:
- Converting word to lower case (NLP->nlp).
- **Q.Why Lower Casing?**
    - Words like Book and book mean the same,
    - When not converted to the lower case those two are represented as two different words in the vector space model (resulting in more dimension).
    - Higher the dimension, more computation resources are required.


## Stemming:
- Converting to the words or tokens to their root word. 
- The root word might not make sense
- It is based on algorithm

Note: Stemming is not available in Spacy library.

## Lemmatization:
- Lemmatization is the process of converting a word to its base form.
- For example, lemmatization would correctly identify the base form of caring to care
- Lemmatization can be carried out using the attribute `token.lemma_`
- It is search-based algorithm

## POS Tagging:
- Parts-of-speech tagging is the process of tagging words in textual input with their appropriate parts of speech.
- This is one of the core feature loaded into the pipeline.
- POS tag can be accessed using `token.pos_`


## Named Entity Recognition
- It is the process of detecting the named entities such as the person name, the location name, the company name, the quantities and the monetary value.
- We can find the named entity using spaCy `ents` attribute class.
- `entity.text` and `entity.label`
- Entity attributes details