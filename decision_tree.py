from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext

# Load and parse the data
def parsePoint(line):
    values = [float(x) for x in line.split(',')]
    return LabeledPoint(values[0], values[1:])

# Load and parse the data file into an RDD of LabeledPoint.
sc = SparkContext()
trainingData = sc.textFile("hw4/training_data_1987_2007.txt")
parsed_trainingData = trainingData.map(parsePoint)

testData = sc.textFile("hw4/test_data.txt")
parsed_testData = testData.map(parsePoint)

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainClassifier(parsed_trainingData, numClasses=2, categoricalFeaturesInfo={},impurity='gini', maxDepth=5, maxBins=32)

# Evaluate model on test instances and compute test error
predictions = model.predict(parsed_testData.map(lambda x: x.features))
labelAndPred = parsed_testData.map(lambda lp: lp.label).zip(predictions)

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
sc.parallelize(["Accuracy", Accuracy, "TP", TP, "FN", FN, "FP", FP, "TN", TN]).saveAsTextFile("DStree_result")

# Save and load model
model.save(sc, "target/tmp/myDecisionTreeClassificationModel")
sameModel = DecisionTreeModel.load(sc, "target/tmp/myDecisionTreeClassificationModel")