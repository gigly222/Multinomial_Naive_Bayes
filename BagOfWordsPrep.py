import collections, re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer #Good stemmer for english language
import os
import io

#This class contains all of the methods to build your bag of words model from your data. Some of these methods
#clean the data by removing stop words such as "is" and "the" and stemming the words closer to their latin roots.

# Get the bag of words model
def bagOfWordsModel(arrayOfAllDocs):
    bagsofwords = [ collections.Counter(re.findall(r'\w+', txt)) for txt in arrayOfAllDocs]
    return bagsofwords

# For each paragraph, turn them into word tokens. Remove stop words and apply stemming.
def prepForBagOfWords(line):
    splitLine = tokenize(line)
    return splitLine

# Removes stop words
def removeStop(line):
    letters_only = re.sub("[^a-zA-Z]", " ", line)  # No numbers
    line = letters_only.rstrip('\n')
    cleanline = line.rstrip('\r\n')
    lowerCase = cleanline.lower() #Convert to lower case so words can be compared
    wordArray= re.split("\s+", lowerCase) #Split on white space
    filtered_words = [word for word in wordArray if word not in stopwords.words('english')]
    onlyImportWords =' '.join(filtered_words) #Turn array back to string
    return (onlyImportWords)

# Stems words on english roots
def stemming(line):
    snowball_stemmer = SnowballStemmer('english')
    splitLine = line.split(" ")
    wordArray = []
    for w in splitLine:
        wordArray.append(snowball_stemmer.stem(w))
    return wordArray

# Function to break text into "tokens", lowercase them, remove stopwords, and stem them
def tokenize(text):
    # Apply stop words filter
    sentenceNoStopWords = removeStop(text)
    # Apply stemming
    stemmedSentence = stemming(sentenceNoStopWords)
    return stemmedSentence

#Read in documents from Directories
def readInDocuments(docPath):

    targetLables = []
    counterForLables = 0
    numOfDocumentsRead  = 0
    ArrayOfDocsInClassCount = []

    # Initialize an empty list to hold all documents without stopwords and stemming applied
    ArrayAllDocs = [];

    for directory in os.listdir(docPath):
        eachTrainDir = docPath + "/" + directory

        print("Reading in this Directory : ", directory)
        docsInClassCounter = 0;
        for file in os.listdir(eachTrainDir):
            docsInClassCounter += 1
            numOfDocumentsRead += 1
            #Reset for next file
            document = []
            pay_attention = False
            for line in io.open(eachTrainDir + "/" + file, encoding="ISO-8859-1"):
                line = line.replace(">", "").replace(">>", "")
                if(pay_attention):
                    wordList = prepForBagOfWords(line)
                    for word in wordList:
                        if len(word) != 0:
                            document.append(word)
                if line.__contains__("Lines:"):
                    pay_attention = True
            stringToPrint = " ".join(document)
            ArrayAllDocs.append(stringToPrint)
            targetLables.append(counterForLables)
            # Append total number of documents counted in that class (directory)
        ArrayOfDocsInClassCount.append(docsInClassCounter)
        counterForLables += 1

    return ArrayAllDocs, numOfDocumentsRead, ArrayOfDocsInClassCount, targetLables

#Wraps up document vectors with their associated target lables
def getArrayofVectorsForAllClasses(bagOfWords, targetLables):
    arraysOfDocPerClass = []  # rows the columns
    testLabel = 0
    docsPerClass = []
    for document, label in zip(bagOfWords, targetLables):
        singleDoc = (document, label)# Tuple of feature vector with matching label. Each one is a document in our training data
        if (label == testLabel):
            docsPerClass.append(singleDoc)  # Array of tuple Doc, Vec and Lable
        else:
            arraysOfDocPerClass.append(docsPerClass)
            docsPerClass = []
            testLabel += 1
            docsPerClass.append(singleDoc)

    arraysOfDocPerClass.append(docsPerClass)
    return arraysOfDocPerClass

