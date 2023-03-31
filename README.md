# TF-IDF Search Engine

This is a command line tool that allows a user to query a collection of documents. 
The query and similarity are done by TF-IDF similarity measures.

# How to Run

Your folder should look like this before running the code:

```
../Tf-Idf_search_engine/
├── LICENSE
├── nyt199501.xml
├── nytsmall.xml
├── README.md
├── softwareAssignment.py
└── stemming
    ├── lovins.py
    ├── paicehusk.py
    ├── porter2.py
    ├── porter.py
    └── __pycache__
        └── porter2.cpython-38.pyc

3 directories, 10 files
```

##  Before running the code
- Move the folder with the provided stemming application to this folder;
- Move the desired xml files to this folder;

## Cli Help Window

```
usage: softwareAssignment.py [-h] [-collection COLLECTION] [-create CREATE]

Command line tool to run a Query on a Collection of documents using TF-IDF
similarity search

optional arguments:
  -h, --help            show this help message and exit
  -collection COLLECTION
                        Name of the collection without file extension
  -create CREATE        True = creates the index files, False=reads from index
                        file
```

To run the application from the terminal use the following pattern.

```
python softwareAssignment.py -collection <collection name> -create <True or False>
```

by default, the ```-create``` argument takes ```false```, since, from a user perspective, it would be used more often, so for a first run, use 

```
python softwareAssignment.py -collection <collection name> -create True
```

