import sys
import collections
from BagOfWordsPrep import readInDocuments, bagOfWordsModel, getArrayofVectorsForAllClasses
from NaiveBayes import multinomailNaiveBayes, testNaiveBayes
from Mesaurements import accuracyTest

#Multinomial Naive Bayes
if __name__ == "__main__":

    # check to see if argument input is the right length
    if len(sys.argv) != 3:
        sys.exit(0)

    trainDocPath = sys.argv[1] #Path to training documents
    testDocPath = sys.argv[2]  #Path to testing documents

    # ***************************
    #Train Naive Bayes Classifier
    #***************************

    # Read in clean documents
    print("Getting Training Data...")
    arrayAllDocsWords, numOfDocumentsRead, arrayOfDocsInClassCount, targetLables = readInDocuments(trainDocPath)

    # fit bag of words model training data
    bagOfWords = bagOfWordsModel(arrayAllDocsWords)
    sumbags = sum(bagOfWords, collections.Counter())
    numOfDistinctWords = len(sumbags)
    documentsInClassArray = getArrayofVectorsForAllClasses(bagOfWords, targetLables)

    #Build multinomial Naive Bayes Model
    priorArray, condProbArray, vocabIndexMap= multinomailNaiveBayes(bagOfWords, numOfDocumentsRead, arrayOfDocsInClassCount, documentsInClassArray)

    # ***************************
    #Test Naive Bayes Classifier
    #***************************

    print("Getting Testing Data...")

    #Read in clean test documents
    arrayAllTestDocsWords, numOfTestDocumentsRead, arrayOfTestDocsInClassCount, targetTestLables, = readInDocuments(testDocPath)

    # fit bag of words model for testing data
    bagOfWordsTest = bagOfWordsModel(arrayAllTestDocsWords)
    documentsInClassArray = getArrayofVectorsForAllClasses(bagOfWordsTest, targetTestLables)

    # Test multinomial Naive Bayes Model
    predictedLabels = testNaiveBayes(bagOfWordsTest, arrayOfTestDocsInClassCount, priorArray, condProbArray, vocabIndexMap)

    # ***************************
    # Accuracy
    # ***************************
    accuracyTest(targetTestLables, predictedLabels)
