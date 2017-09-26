
from keras.models import Sequential, h5py, load_model
import matplotlib
import numpy
import h5py
from keras.layers import Dense,Dropout
from sklearn.preprocessing import StandardScaler, normalize
import time
import urllib.request

testSetFilenameCollection = [
							 # "learning_set/nnDataSet1_2.csv",
							 # "learning_set/nnDataSet2_1.csv",
							 # "learning_set/nnDataSet2_2.csv",
							 # "learning_set/nnDataSet3_1.csv",
							 "learning_set/nnDataSet3_2.csv",
							 "test_set/nnDataSet1_3.csv",
							 # "test_set/nnDataSet1_4.csv",
							 # "test_set/nnDataSet1_5.csv",
							 # "test_set/nnDataSet1_6.csv",
							 # "test_set/nnDataSet1_7.csv",
							 "test_set/nnDataSet2_3.csv",
							 # "test_set/nnDataSet2_4.csv",
							 # "test_set/nnDataSet2_5.csv",
							 # "test_set/nnDataSet2_6.csv",
							 # "test_set/nnDataSet2_7.csv",
							 # "test_set/nnDataSet3_3.csv"
							 ]

model = load_model('FullTrainingSet_epochs2100.h5')


##########################################################################################################################
def PredictForTestSetCollection(filenames):
	for filename in filenames:
		predictForTestSet(filename)
MaxYPredict = 0
datasetPredict = []
XPredict = []
def predictForTestSet(filename):
	global datasetPredict,XPredict,MaxYPredict,model
	datasetPredict = numpy.loadtxt(filename, delimiter=",")

	totalSizePredict = len(datasetPredict)-2
	# print(totalSizePredict)
	# numpy.random.shuffle(dataset)
	XPredict = numpy.zeros((totalSizePredict,6))
	for j in range(totalSizePredict):
		XPredict[j][0] = datasetPredict[j,0]
		XPredict[j][2] = datasetPredict[j,1]
		XPredict[j][4] = datasetPredict[j,2]

	for j in range(1,totalSizePredict):
		XPredict[j][1] = XPredict[j-1][0]
		XPredict[j][3] = XPredict[j-1][2]
		XPredict[j][5] = XPredict[j-1][4]
	YPredict = datasetPredict[:,3]
	# print (X)
	# print (Y)

	MaxPredict = numpy.zeros(len(XPredict[0]))
	XPredict = XPredict.astype(float)
	for i in range(len(XPredict[0])):
		MaxPredict[i] = 0
		for j in range(len(XPredict)):
			if(MaxPredict[i] < XPredict[j][i]):
				MaxPredict[i] = XPredict[j][i]
		# print ("Max = ", MaxPredict[i])
		for j in range(len(XPredict)):
			XPredict[j][i] = XPredict[j][i] / MaxPredict[i]

	YPredict = YPredict.astype(float)

	MaxYPredict = 0
	for i in range(len(YPredict)):
		if(MaxYPredict < YPredict[i]):
			MaxYPredict = YPredict[i]
	# print ("MaxYPredict = ", MaxYPredict)
	for i in range(len(YPredict)):
		YPredict[i] = YPredict[i] / MaxYPredict

	errorSumPredict=0
	totalReadingsPredict = 0
	for j in range((totalSizePredict)):
		a = model.predict(numpy.array(XPredict[j]).reshape(-1,6))
		if(YPredict[j] == 0):
			continue
		error_percPredict = abs(((YPredict[j]-a)*100)/YPredict[j])
		# predictedRUL = a*MaxYPredict
		# predictedRUL = datasetPredict[j, 0]/(1.0-predictedRUL)
		# print("a = ", a, " YPredict = ", YPredict[j])
		# print("PredictedRUL = ", predictedRUL, " Real RUL = ", datasetPredict[j, 3], 
		# 	"Perc error = ", (predictedRUL - datasetPredict[j, 3]) / datasetPredict[j, 3] * 100)
		# print("error_perc [", j, "] = ", error_percPredict)
		errorSumPredict = errorSumPredict+error_percPredict
		totalReadingsPredict += 1
	print ("For Test Set - ", filename, ", Avg Error in Test Data Set = ", (errorSumPredict/(totalReadingsPredict)))

def predictRUL(inputReadings):
	predictedRUL = model.predict(numpy.array(inputReadings).reshape(-1,6))
	predictedRUL = inputReadings[0]/(1.0-predictedRUL)
	return predictedRUL
# inputReadings[0]/predictedRUL = (1-a)
def importTestSet(filename):
	global datasetPredict,XPredict,MaxYPredict,model
	datasetPredict = numpy.loadtxt(filename, delimiter=",")

	totalSizePredict = len(datasetPredict)-2
	# print(totalSizePredict)
	# numpy.random.shuffle(dataset)
	XPredict = numpy.zeros((totalSizePredict,6))
	for j in range(totalSizePredict):
		XPredict[j][0] = datasetPredict[j,0]
		XPredict[j][2] = datasetPredict[j,1]
		XPredict[j][4] = datasetPredict[j,2]

	for j in range(1,totalSizePredict):
		XPredict[j][1] = XPredict[j-1][0]
		XPredict[j][3] = XPredict[j-1][2]
		XPredict[j][5] = XPredict[j-1][4]
	YPredict = datasetPredict[:,3]
	# print (X)
	# print (Y)

	MaxPredict = numpy.zeros(len(XPredict[0]))
	XPredict = XPredict.astype(float)
	for i in range(len(XPredict[0])):
		MaxPredict[i] = 0
		for j in range(len(XPredict)):
			if(MaxPredict[i] < XPredict[j][i]):
				MaxPredict[i] = XPredict[j][i]
		# print ("Max = ", MaxPredict[i])
		for j in range(len(XPredict)):
			XPredict[j][i] = XPredict[j][i] / MaxPredict[i]

	YPredict = YPredict.astype(float)

	MaxYPredict = 0
	for i in range(len(YPredict)):
		if(MaxYPredict < YPredict[i]):
			MaxYPredict = YPredict[i]
	# print ("MaxYPredict = ", MaxYPredict)
	for i in range(len(YPredict)):
		YPredict[i] = YPredict[i] / MaxYPredict


##########################################################################################################################

PredictForTestSetCollection(testSetFilenameCollection)



timePassed = 6
time.sleep(2)
print ("LEN = ", len(testSetFilenameCollection))
predictedRULs = [0, 0, 0]
healths = [0, 0, 0]
while True:
# for i in range(len(datasetPredict)-2):
	# inputForNN = [datasetPredict[timePassed][0], datasetPredict[timePassed-1][0], 
	# 		      datasetPredict[timePassed][1], datasetPredict[timePassed-1][1], 
	# 		      datasetPredict[timePassed][2], datasetPredict[timePassed-1][2]]
	for i in range(len(testSetFilenameCollection)):
		importTestSet(testSetFilenameCollection[i])
		if(len(datasetPredict)-2 < timePassed):
			predictedRULs[i] = 0
			continue
		realRUL = datasetPredict[timePassed][3]
		# predictedRUL = MaxYPredict*(model.predict(numpy.array([0.00125628, 0, 0.07505039, 0.07033415, 0.06912259, 0.06125143]).reshape(-1,6)))
		# predictedRUL = MaxYPredict*(model.predict(numpy.array(inputForNN).reshape(-1,6)))
		health = model.predict(numpy.array(XPredict[timePassed][:]).reshape(-1,6))
		predictedRUL = MaxYPredict*health
		predictedRULs[i] = predictedRUL
		healths[i] = health[0][0]
	print("Time = ",timePassed," PredictedRULs = ", predictedRULs[0], " , ", predictedRULs[1], " , ", predictedRULs[2], "")
	req1 = "https://siemenspdm.000webhostapp.com/bearingStats.php?queryType=insert&bID=b1&data={\"health\":\"" + str(healths[0]) + "\",\"rul\":\"" + str(predictedRULs[0][0][0]) + "\",\"day\":\"1\",\"month\":\"June\",\"year\":\"2017\"}";
	req2 = "https://siemenspdm.000webhostapp.com/bearingStats.php?queryType=insert&bID=b2&data={\"health\":\"" + str(healths[1]) + "\",\"rul\":\"" + str(predictedRULs[1][0][0]) + "\",\"day\":\"15\",\"month\":\"January\",\"year\":\"2017\"}";
	req3 = "https://siemenspdm.000webhostapp.com/bearingStats.php?queryType=insert&bID=b3&data={\"health\":\"" + str(healths[2]) + "\",\"rul\":\"" + str(predictedRULs[2][0][0]) + "\",\"day\":\"30\",\"month\":\"4\",\"December\":\"2016\"}";
	print(req1)
	print(predictedRULs[0][0][0])
	urllib.request.urlopen(req1);
	urllib.request.urlopen(req2);
	urllib.request.urlopen(req3);
	timePassed += 1


# set1 = [ 10, 0, 0.24583]
# 0	0.23878	2.9138	6110
# 10	0.24583	2.9557	6100

# print("PredictedRUL = ", predictRUL(set1[0:5]), " Real RUL = ", set1[6])
# print("Perc error = ", (predictRUL(set1[0:5]) - set1[6]) / set1[6] * 100)

# predictedRUL = model.predict(numpy.array().reshape(-1,6))