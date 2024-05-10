import re
import emoji
import string
import spacy

from nltk.stem.porter import PorterStemmer
# pip install nltk
# pip install emoji


class BasicCleaning:

    def lower_casing(self, text):
        """
        Convert the given text into lower case and return the converted one.

        Arguments:
        text (string) : The raw text that needs to be converted to lower case

        Returns:
        lower_text (string): The lower case converted form oflower text
        """

        lower_text = text.lower()
        return lower_text

    def remove_html(self, text):
        """
        Detect and remove all the html tags from the string

        Arguments:
        text(string): The raw text from which the html tags needs to be removed

        Returns:
        text_after_html(string) : The text from which html tags are removed.
        """

        html_pattern = re.compile("<.*?>")
        text_after_html = html_pattern.sub(r"", text)
        return text_after_html

    def remote_url(self, text):
        """
        Detect the urls in text, remove them and return the final text

        Arguments:
        text(string): The raw text form

        Returns:
        text_after_url(string): text which doesn't contain any sorts fo url
        """

        url_pattern = re.compile(r"https?://\S+|www\. \S+")
        text_after_url = url_pattern.sub(r"", text)
        return text_after_url

    def remove_emoji(self, text, replace_with_meaning=True):
        """
        Detect any sorts of emoji and remove (or replace with meanings) from the original text

        Arguments:
        text(string): raw text that have emojis
        replace_with_meaning(bool) : whether emojis are removed or replaced

        Returns:
        text_after_emoji(string): text after removing the emojis fromt the original text
        """

        text_after_emoji = emoji.demojize(text)
        return text_after_emoji

    def remove_punctuation(self, text):
        """
        Detects punctuation marks and remove them from the original text

        Arguments:
        text(string): raw text that have punctuations

        Returns:
        text_after_punc(string): text after punctuations are removed.
        """

        exclude_punc = string.punctuation
        for char in exclude_punc:
            text = text.replace(char, "")

        text_after_punc = text
        return text_after_punc


class BasicPreprocessing:

    def __init__(self):
        self.basic_clean = BasicCleaning()
        self.nlp = spacy.load("en_core_web_sm")

    def tokenize_words(self, text):
        """
        Accepts string and convert into smaller tokens on the basis of words.

        Arguments:
        text(string) : raw text most probably after basic cleaning is done

        Returns:
        word_tokens(list) : list of word tokens
        """
        
        doc = self.nlp(text)
        word_tokens = [token.text for token in doc]

        return word_tokens

    def tokenize_sentence(self, text):
        """
        Accepts string and convert into smaller tokens on the basis of sentence.

        Arguments:
        text(string) : raw text most probably after basic cleaning is done

        Returns:
        sent_tokens(list) : list of word tokens
        """

        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]

        return sentences

    def stop_word_removal(self, text):
        """
        Removes the stop words from the text and return the filtered text

        Arguments:
        text(string): raw text

        Returns:
        filtered_tokens(list): list of tokens from which stop words have been removed.
        """

        stopwords = self.nlp.Defaults.stop_words
        doc = self.nlp(text)

        filtered_tokens = [token.text for token in doc if token.text not in stopwords]

        return filtered_tokens

    def stemming(self, text):
        """
        Tokenize the given string on the basis of word tokenization
        and reduce each tokens back to root word algorithmly

        Arguments:
        text(string) : raw text for stemming purpose

        Returns:
        stem_tokens(list) : list of stemmed tokens
        """
        # Tokenize:
        word_tokens = self.tokenize_words(text)

        # Instantiated PorterStemmer Algorithm
        ps = PorterStemmer()
        stem_tokens = [ps.stem(word) for word in word_tokens]
        return stem_tokens

    def lemmatization(self, text):
        """
        Tokenize the given string on the basis of word tokenization
        and reduce each tokens back to root word by searching language dictionary.

        Arguments:
        text(string) : raw text for lemmatization purpose

        Returns:
        lemma_tokens(list) : list of lemmatized tokens
        """
        lemma_tokens= []
        doc = self.nlp(text)

        for token in doc:
            lemma_tokens.append(token.lemma_)
            
        return lemma_tokens


class AdvancedPreprocessing:

    def __init__(self):
        self.basic_preprocess = BasicPreprocessing()

    def pos_tagging(self, text):
        """
        Find out the corresponding Part of Speech Tag for each token in the text

        Arguments:
        text (string): raw text

        Returns:
        tags (list) : list of tupples where 1st element is token and 2nd element is appropriate POS tag.
        """
        tags = []
        doc = self.nlp(text)

        for token in doc:
            tags.append((token.text, token.pos_))

        return tags

    def named_entity_recognizer(self, text, binary=False):
        """
        Find out the words that are named entity like people, location, org etc.

        Arguments:
        text(string) : raw text
        binary (bool): Flags whether to give entity a label
                        True => doesn't give label to named entity
                        False => gives the label to named entity (default)

        Returns:
        named_entities(list): list of tuples containing all the named entities and corresponsing label either as binary or types of NE
        """

        named_entities = []
        doc = self.nlp(text)

        for entity in doc.ents:
            named_entities.append((entity.text, entity.pos_))

        return named_entities