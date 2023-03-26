'''
Introduction to Python Programming (aka Programmierkurs I, aka Python I)
Software Assignment
'''

import sys
import string, math
from lxml import etree
from stemming.porter2 import stem
from collections import Counter

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
        if create == True:
            #  Read files into an lxml etree object
            tree = etree.parse(collectionName+".xml")
            root = tree.getroot()

            # A dictionary where key = doc id, value = doc as a list of word ids
            corpus = {}
            # A dictionary where key is a word and the value is the nr of docs
            # it appears in
            vocab = {}
            
            
            
            for doc in root.iter('DOC'):
                # initializing and reseting the document text string
                doc_txt = ""
                for element in doc.iter('HEADLINE', "P", "TEXT"):
                    doc_txt += element.text.lower()
                # Removing punctuation
                doc_txt = doc_txt.translate(str.maketrans('', '', string.punctuation))
                # Applying stemming to words
                doc_txt = list(map(stem, doc_txt.split()))

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
            # number of documents in the corpus
            n_docs = len(corpus)

            # IDF
            self.idf_dict = {}
            with open(collectionName + '.idf', 'w') as file:
                for term, count in vocab.items():
                    idf = math.log(n_docs/count)
                    self.idf_dict[term] = idf
                    file.write('{}\t{}\n'.format(term, idf))
            

            # TF
            self.tf_dict = {}
            with open(collectionName+".tf", "w") as file:
                for key, text in corpus.items():
                    x = Counter(text)
                    # Returns a list of tuples containing pair of values where
                    # the most frequent term in index 0 and its count in index 1
                    # To get the 
                    max_occurence = x.most_common(1)[0][1]

                    for word, count in dict(sorted(x.items())).items():
                        tf = count/max_occurence
                        self.tf_dict[key] = {word,tf}
                        file.write('{}\t{}\t{}\n'.format(key, word, tf))
            
            # TF-IDF
            tf_idf = {}
            for doc_id, text in corpus.items():
                

            # TODO: Create TF; save to <collectionName>.tf
            # TODO: Create IDF; save to <collectionName>.idf
        
        else:
            with open(collectionName+'.tf', 'r') as file:
                self.tf = file.read()
            with open(collectionName+'.idf', 'r') as file:
                self.idf = file.read()

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
        pass
        
    def executeQueryConsole(self):
        '''
        When calling this, the interactive console should be started,
        ask for queries and display the search results, until the user
        simply hits enter.
        '''
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
    searchEngine = SearchEngine("nytsmall", create=True)
    print(searchEngine.executeQuery(['hurricane', 'philadelphia']))
