'''
Introduction to Python Programming (aka Programmierkurs I, aka Python I)
Software Assignment
'''

import sys
import string
import math
import numpy as np
from lxml import etree
from stemming.porter2 import stem
from collections import Counter

from pprint import pprint


def pre_process(in_txt: str) -> list:
    """Receives a string of text, removes the punctuation, turns text into 
    lowercase and performs stemming. Returns the input string as a list of 
    tokens
    in_txt:<str> - text to be processed in string
    txt:<list of str> - processed text as a bag-of-words/ list of tokens
    """
    in_txt = in_txt.translate(str.maketrans(
        '', '', string.punctuation)).lower()
    txt = list(map(stem, in_txt.split()))
    return txt


def tf_func(text_list: list) -> dict:
    """ Returns a dictionary containing the word the its term frequency value
    as a key value par in a dict

    text_list:<list> -: input text as a list
    return_dict:<dict> -: key:value -> word:tf
    """
    return_dict = {}
    # used Counter dictionary from the standar python lib
    x = Counter(text_list)
    # uses the most_common() method to the get the top1 most common token
    max_occurence = x.most_common(1)[0][1]
    # Sorts the dict alphabetically
    x = dict(sorted(x.items()))
    # We iterate over the dict and get the tf value for every type in the text
    for word, count in x.items():
        tf = count/max_occurence
        return_dict.update({word: tf})
    return return_dict


def idf_func(vocab: dict, corpus_len: int) -> dict:
    """Function to calculate IDF over vocab dict. The vocab dict is a dict
    that contains the doc IDs as keys and the text as a list of tokens in its
    values

    vocab:<dict>        -: consists of doc_id:list of tokens in the doc
    corpus_len:<int>    -: number of documents in the corpus
    idf_dict:<dict>     -: word:idf pairs
    """
    idf_dict = {}
    for term, count in vocab.items():
        idf = math.log(corpus_len/count)
        idf_dict[term] = idf
    return idf_dict


class SearchEngine:

    def __init__(self, collectionName, create):
        '''
        Initialize the search engine, i.e. create or read in index. If
        create=True, the search index should be created and written to
        files. If create=False, the search index should be read from
        the files. The collectionName points to the filename of the
        document collection (without the .xml at the end). Hence, you
        can read the documents from <collectionName>.xml, and should
        write / read the idf index to / from <collectionName>.idf, and
        the tf index to / from <collectionName>.tf respectively. All
        of these files must reside in the same folder as THIS file. If
        your program does not adhere to this "interface
        specification", we will subtract some points as it will be
        impossible for us to test your program automatically!
        '''
        self.tf_dict = {}
        self.idf_dict = {}
        self.tf_idf = {}

        if create == True:
            print("Creating index files")
            # Read files into an lxml etree object
            tree = etree.parse(collectionName+".xml")
            # Gets root node so we can iterate over it
            root = tree.getroot()

            # A dictionary where key = doc id, value = doc as a list of word ids
            corpus = {}
            # A dictionary where key is a word and the value is the nr of docs
            # it appears in
            vocab = {}

            for doc in root.iter('DOC'):
                # initializing the document text string, adds the content from
                # the tags "HEADLINE", "P" and "TEXT"
                doc_txt = ""
                for element in doc.iter('HEADLINE', "P", "TEXT"):
                    doc_txt += element.text.lower()

                doc_txt = pre_process(doc_txt)

                # Creating Vocab and increasing the count if word exists
                for word in set(doc_txt):
                    if word not in vocab:
                        vocab[word] = 1
                    else:
                        vocab[word] += 1

                # Appending the document to the corpus dict
                corpus[doc.attrib['id']] = doc_txt
            # Sorting the vocabulary dict on keys (words)
            vocab = dict(sorted(vocab.items()))

            # Sorting thecorpus dict on keys
            corpus = dict(sorted(corpus.items()))

            # IDF
            self.idf_dict = idf_func(vocab, len(corpus))
            # Write .idf files
            with open(collectionName + '.idf', 'w') as file:
                for term, idf in self.idf_dict.items():
                    file.write('{}\t{}\n'.format(term, idf))

            # TF
            # Iterate over the corpus creating a dictionary where key->doc_id
            # value another dict with the tf values for the document
            for doc_id, text in corpus.items():
                self.tf_dict[doc_id] = tf_func(text)
            # Write .tf files
            with open(collectionName+".tf", "w") as file:
                for doc_id, w_tf_pair in self.tf_dict.items():
                    for word, tf in w_tf_pair.items():
                        file.write('{}\t{}\t{}\n'.format(doc_id, word, tf))

            print("Done creating indexes")

        if create == False:
            print('Reading indexes from files')
            # Read .tf file of specified corpus
            with open(collectionName+'.tf', 'r') as file:
                for line in file.readlines():
                    doc_id, term, tf = line.split('\t')
                    # Converts the tf value from str to float
                    tf = float(tf.strip())

                    # Check if the corresponding doc already exists in the
                    # tf_dict if it doesn't, create the empty dict and update it
                    # with the current values for term and tf
                    if doc_id not in self.tf_dict.keys():
                        self.tf_dict[doc_id] = {}
                        self.tf_dict[doc_id].update({term: tf})
                    else:
                        # add the term and its tf value to the corresponding dict
                        # for this document
                        self.tf_dict[doc_id].update({term: tf})

            # Read .idf file of specified corpus
            with open(collectionName+'.idf', 'r') as file:
                for line in file.readlines():
                    term, idf = line.split('\t')

                    self.idf_dict[term] = float(idf.strip())
        # TF-IDF
        # Created a word:index dict to convert the words into indices in the
        # tf-idf vectors
        self.word_to_ix = {key: value for key, value in zip(self.idf_dict.keys(),
                                                            range(len(self.idf_dict)))}

        # Iterates over the tf_dict
        for doc_id, term_tf_pair in self.tf_dict.items():
            # Initializes the vector for the current document with zeros
            self.tf_idf[doc_id] = np.zeros(len(self.idf_dict))

            # iterates over the tf values in the term_tf_pair dicr
            for term, tf in term_tf_pair.items():
                # pass tf-idf value to tf_idf dict, acess the correct index
                # by the word_to_ix dict
                self.tf_idf[doc_id][self.word_to_ix[term]
                                    ] = tf * self.idf_dict[term]
        pass

    def executeQuery(self, queryTerms):
        '''
        Input to this function: List of query terms

        Returns the 10 highest ranked documents together with their
        tf.idf-sum scores, sorted score. For instance,

        [('NYT_ENG_19950101.0001', 0.07237004260325626),
         ('NYT_ENG_19950101.0022', 0.013039249597972629), ...]

        May be less than 10 documents if there aren't as many documents
        that contain the terms.
        '''
        # pre process query:
        query_str = " ".join(queryTerms)
        query = pre_process(query_str)

        # Query TF:
        query_tf = tf_func(query)

        # Query TF-IDF
        query_tf_idf = np.zeros(len(self.idf_dict))

        for term in query:
            try:
                query_tf_idf[self.word_to_ix[term]
                             ] = query_tf[term] * self.idf_dict[term]
            except KeyError:
                return []

        # Calculate similarities by iterating over the tf_idf dict
        sim_dict = {}
        for doc_id, value in self.tf_idf.items():
            top = np.dot(query_tf_idf, value)
            bottom = np.linalg.norm(query_tf_idf) * np.linalg.norm(value)
            sim = top/bottom
            if sim == 0:
                continue
            else:
                sim_dict.update({doc_id: sim})

        sim_dict = sorted(sim_dict.items(), reverse=True,
                          key=lambda item: item[1])
        return sim_dict[:10]

    def executeQueryConsole(self):
        '''
        When calling this, the interactive console should be started,
        ask for queries and display the search results, until the user
        simply hits enter.
        '''
        query = input("Please enter query, terms separated by whitespace: ")
        while query != "":
            result_list = self.executeQuery(query.split())
            if len(result_list) == 0:
                print("Sorry, I didn’t find any documents for this term.")
            else:
                print("I found the following documents:")
                for doc_id, similarity in result_list:
                    print("{} ({})".format(doc_id, similarity))

            query = input(
                "Please enter query, terms separated by whitespace: ")

        pass


if __name__ == '__main__':
    '''
    write your code here:
    * load index / start search engine
    * start the loop asking for query terms
    * program should quit if users enters no term and simply hits enter
    '''
    # Example for how we might test your program:
    # Should also work with nyt199501 !
    searchEngine = SearchEngine("nyt199501", create=True)
    searchEngine.executeQueryConsole()
    # print(searchEngine.executeQuery(['hurricane', 'philadelphia']))
    # searchEngine2 = SearchEngine("nytsmall", create=False)
    # print(searchEngine2.executeQuery(['hurricane', 'philadelphia']))

    # print(searchEngine.executeQuery(['atermnotinvocab']))

    # searchEngine3 = SearchEngine("nytsmall", create=True)
    # print(searchEngine3.executeQuery(['hurricane', 'philadelphia']))
    # searchEngine4 = SearchEngine("nytsmall", create=False)
    # print(searchEngine4.executeQuery(['hurricane', 'philadelphia']))
