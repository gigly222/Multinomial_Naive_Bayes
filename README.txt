-README

**********************************************
Multinomial Naive Bayes Text Classification
**********************************************

-Written in Python 3.5.2.
-Pycharm2016.2.3 was used for development.
-I used nltk to remove stop words and stem my documents. The following link contains instructions on how to downlaod and install: http://www.nltk.org/install.html
-The bag of words model was created using a collections object from python. 
-The multnomial naive bayes model and testing was written from scratch by me with no ML libraries. 
-To clarify, No machine learning libraries were used in the creation of the Naive Bayes program. 

The 20 newsgroups dataset was used following this link: 
http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz

I used the following 5 training and test folders (You can choose any 5 from the website):
comp.graphics
sci.space
rec.autos
sci.med
talk.religion.misc

To run the program, it is important the correct path to the training and test data is provided. 
My directories had the following path structures:
	20news-bydate-train/
		trainDocs/
			comp.graphics  rec.autos  sci.med  sci.space  talk.religion.misc

	20news-bydate-test/
		testDocs/
			comp.graphics  rec.autos  sci.med  sci.space  talk.religion.misc

*The directories such as comp.graphics or rec.autos contain a plethora of text documents of that class.*


*************************************************************************************************
To Run the Multinomial Naive Bayes Program program, type in something following the format below:
*************************************************************************************************

python Main.py "/home/gigly/Documents/machine_learning/bayesDocs/20news-bydate-train/trainDocs" "/home/gigly/Documents/machine_learning/bayesDocs/20news-bydate-test/testDocs"

*************************************************************************************************

*The two arguments are the first path to the directory containing the 5 training class directories, and the second is the path to the directory containing
the 5 testing class directories. 
