from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])

# Load and parse the data file into an RDD of LabeledPoint.
sc = SparkContext()
data = sc.textFile("hw4/training_data_1987_2007.txt")
parsedData = data.map(parsePoint)

# Build the model
model = LogisticRegressionWithLBFGS.train(parsedData)

# Evaluating the model on training data
test_data = sc.textFile("hw4/test_data.txt")
parsed_testData = test_data.map(parsePoint)
labelAndPred = parsed_testData.map(lambda p: (p.label, model.predict(p.features)))

# Prediction Result
Accuracy = labelAndPred.filter(lambda (v, p): v == p).count() / float(parsed_testData.count())
TP = labelAndPred.filter(lambda (v,p): v==1 and p==1).count()
FN = labelAndPred.filter(lambda (v,p): v==1 and p==0).count()
FP = labelAndPred.filter(lambda (v,p): v==0 and p==1).count()
TN = labelAndPred.filter(lambda (v,p): v==0 and p==0).count()

print("*** Test Result ***")
print("Accuracy = " + str(Accuracy))
print("TP = %s " % TP)
print("FN = %s " % FN)
print("FP = %s " % FP)
print("TN = %s " % TN)

# output data
sc.parallelize(["Accuracy", Accuracy, "TP", TP, "FN", FN, "FP", FP, "TN", TN]).saveAsTextFile("LR_result")


# Save and load model
model.save(sc, "myModelPath")
sameModel = LogisticRegressionModel.load(sc, "myModelPath")