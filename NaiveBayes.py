import collections
import math

#This Class contains all the methods needed to build the Multinomial Naive Bayes Model from scratch and test it.

# Get all of the documents in all classes and return in array format
def getAllDocumentsasArray(documentsInClassArray, arrayOfTestDocsInClassCount):
    arrayOfTestDocs = []
    for i in range(len(arrayOfTestDocsInClassCount)):
        for document in documentsInClassArray[i]:
            arrayOfTestDocs.append(document[0])
    return arrayOfTestDocs


# Get total vocabulary of words used in all documents of all classes
def getAllVocabUsed(bagOfWordsModel):
    vocabIndexMap = {}
    sumbags = sum(bagOfWordsModel, collections.Counter())
    for vocabWord in sumbags:
        vocabIndexMap[vocabWord] = vocabWord
    for i,(word) in enumerate(vocabIndexMap):
        vocabIndexMap[word] = i

    return vocabIndexMap #All distinct words in all documents and all classes

# Get total vocabulary of words used in documents of a single class
def concatenateTextOfAllDocsInClass(allDocsInClass):
    concatDocs = []
    concatenatedWordsPerDoc = []

    for tupleDoc in allDocsInClass:
        concatDocs.append(tupleDoc[0])
    sumbagsForClass= sum(concatDocs, collections.Counter())

    for vocabWord in sumbagsForClass:
        concatenatedWordsPerDoc.append(vocabWord)
    return concatenatedWordsPerDoc, sumbagsForClass #All distinct words in a class


#Counts the terms in documents belonging to class c:
def countTokensOfTerm(eachWord, sumbagsForClass): #term t in documents belonging to class c:
    c = collections.Counter(sumbagsForClass)
    wordCount = c[eachWord]
    return wordCount


# Naive Bayes Model
def multinomailNaiveBayes(bagOfWordsModel, numOfDocumentsRead, arrayOfDocsInClassCount, documentsInClassArray):

    print("Traning Naive Bayes Model")
    totalNumDocs = numOfDocumentsRead # N is the total number of documents.
    priorArray = [] #Store all of the priors for all of the classes
    condProbArray = [] #Store all of the conditional probabilities in an array per class

    # Get all vocabulary words in all documents in all classes
    vocabIndexMap = getAllVocabUsed(bagOfWordsModel)

    #For each class
    for i in range(len(arrayOfDocsInClassCount)):
        #Initialize array to hold all of the word counts for each token in each document
        tokenCountArray = []
        runningTotal = 0
        probArray = []
        numDocsInClass = arrayOfDocsInClassCount[i] #numDocsInClass is the number of documents in class c

        #Calculate prior for each class
        prior = numDocsInClass/totalNumDocs
        priorArray.append(prior)

        totalWordsPerClass, sumbagsForClass = concatenateTextOfAllDocsInClass(documentsInClassArray[i]) #Get total vocab used in a class

        for eachWord in vocabIndexMap:
            tokenCount = countTokensOfTerm(eachWord, sumbagsForClass) #count how many times that particular word showed up in the set of documents for a particular class
            tokenCountArray.append(tokenCount + 1)
            runningTotal += ( tokenCount + 1)

        # Calculate Condition Probability for each token
        for eachToken in tokenCountArray:
            calculatedProbability = eachToken/runningTotal
            probArray.append(calculatedProbability)

        condProbArray.append(probArray)
        sum = 0
        for prop in probArray:
            sum += prop
    return priorArray, condProbArray, vocabIndexMap


#Testing Naive Bayes
def testNaiveBayes(bagOfWordsTest, arrayOfTestDocsInClassCount, priorArray, condProbArray, vocabIndexMap):
    print("Testing Naive Bayes Model")
    max = float('inf')
    maxClass = -1
    scoreArray = []

    #For each documents
    for bagOfWordsForDoc in  bagOfWordsTest:
        for i in range(len(arrayOfTestDocsInClassCount)):
            score = math.log(priorArray[i], 2)  # Score : log prior[c]

            for word in bagOfWordsForDoc:# look up word index in vocabIndexMap. Use index in condProbArray for value
                if (word in vocabIndexMap):  # Check to see if word key exists in hashmap
                    index = vocabIndexMap[word]
                    probValueofWord = condProbArray[i][index]

                    # Find how many times that word appeard in document. You will need to add its probability this many times
                    tokenCount = countTokensOfTerm(word, bagOfWordsForDoc)
                    score += (tokenCount * math.log(probValueofWord, 2)) #Need to multiply by the amount of times the token appeared in the document

            #Find max score, append this to score Array.
            if(max > abs(score)):
                max = abs(score)
                maxClass = i

        scoreArray.append(maxClass)
        # Reset for next document
        max = float('inf')
        maxClass = -1

    return scoreArray

