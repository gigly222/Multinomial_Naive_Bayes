
# Determine the accuracy of your model on your test data
def accuracyTest(actualLabels, predictedLables):
    totalValues = 0
    totalRight = 0
    for actual, predicted in zip(actualLabels, predictedLables):
        totalValues += 1
        if(actual == predicted):
            totalRight += 1

    accuracy = (totalRight / totalValues) * 100
    print("Accuracy is %.2f" % round(accuracy,2) , "%")
    return accuracy
