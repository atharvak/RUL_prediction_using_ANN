# %% 1 
# Package imports 
from keras.models import Sequential, h5py, load_model
import matplotlib
import numpy
import h5py
from keras.layers import Dense,Dropout
from sklearn.preprocessing import StandardScaler, normalize
 # load pima indians dataset

epochsPerLoop = 200
epochLoops = 50
fittingVerbose = 1

# datasetLearning = numpy.loadtxt("learning_set/nnDataSet2_2.csv", delimiter=",")

learningSetFilenameCollection = [
								 "learning_set/nnDataSet1_1.csv",
								 # "learning_set/nnDataSet1_2.csv",
								 # "learning_set/nnDataSet2_1.csv",
								 # "learning_set/nnDataSet2_2.csv",
								 # "learning_set/nnDataSet3_1.csv",
								 # "learning_set/nnDataSet3_2.csv",
								 ]

testSetFilenameCollection = [
							 "learning_set/nnDataSet1_2.csv",
							 "learning_set/nnDataSet2_1.csv",
							 "learning_set/nnDataSet2_2.csv",
							 "learning_set/nnDataSet3_1.csv",
							 "learning_set/nnDataSet3_2.csv",
							 "test_set/nnDataSet1_3.csv",
							 "test_set/nnDataSet1_4.csv",
							 "test_set/nnDataSet1_5.csv",
							 "test_set/nnDataSet1_6.csv",
							 "test_set/nnDataSet1_7.csv",
							 "test_set/nnDataSet2_3.csv",
							 "test_set/nnDataSet2_4.csv",
							 "test_set/nnDataSet2_5.csv",
							 "test_set/nnDataSet2_6.csv",
							 "test_set/nnDataSet2_7.csv",
							 "test_set/nnDataSet3_3.csv",]



##########################################################################################################################
def PredictForTestSetCollection(filenames):
	for filename in filenames:
		predictForTestSet(filename)

def predictForTestSet(filename):
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
		# print("error_perc [", j, "] = ", error_percPredict)
		errorSumPredict = errorSumPredict+error_percPredict
		totalReadingsPredict += 1
	print ("For Test Set - ", filename, ", Avg Error in Test Data Set = ", (errorSumPredict/(totalReadingsPredict)))



##########################################################################################################################




def createModel():
	model = Sequential()
	model.add(Dense(13, input_dim=6, init = 'normal', activation='relu'))
	# model.add(Dropout(0.2))
	model.add(Dense(10,init = 'normal', activation='relu'))
	# model.add(Dense(3,  init = 'normal', activation='relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(1, init = 'normal'))
	# model.add(Dense(1, init = 'normal',  activation='sigmoid'))


	# Compile model
	model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae'])
	return model



##########################################################################################################################



model = createModel()

def learnFromLearningSetCollection(filenames):
	global model
	datasetLearning = numpy.loadtxt(filenames[0], delimiter=",")
	print ("dataset_shape = ",datasetLearning.shape)
	for h in range(1,len(filenames)):
		# learnFromTestSet(filename)
		datasetLearningTemp = numpy.loadtxt(filenames[h], delimiter=",")
		# datasetLearning = numpy.loadtxt(filenames[h], delimiter=",")
		datasetLearning = numpy.row_stack((datasetLearningTemp, datasetLearning))
		print ("dataset_shape = ",datasetLearning.shape,"\tdatasetTemp_shape = ",datasetLearningTemp.shape)
	totalSizeLearning = len(datasetLearning)-2
	# print(totalSize)
	trainLimitLearning = int(((len(datasetLearning)-2)/100)*100)
	# print(trainLimit)

	numpy.random.shuffle(datasetLearning)

	XLearning = numpy.zeros((totalSizeLearning,6))
	for j in range(totalSizeLearning):
		XLearning[j][0] = datasetLearning[j,0]
		XLearning[j][2] = datasetLearning[j,1]
		XLearning[j][4] = datasetLearning[j,2]
	for j in range(1,totalSizeLearning):
		XLearning[j][1] = XLearning[j-1][0]
		XLearning[j][3] = XLearning[j-1][2]
		XLearning[j][5] = XLearning[j-1][4]
	YLearning = datasetLearning[:,3]
	# print (X)
	# print (Y)

	MaxLearning = numpy.zeros(len(XLearning[0]))
	XLearning = XLearning.astype(float)
	for i in range(len(XLearning[0])):
		MaxLearning[i] = 0
		for j in range(len(XLearning)):
			if(MaxLearning[i] < XLearning[j][i]):
				MaxLearning[i] = XLearning[j][i]
		# print ("Max = ", Max[i])
		for j in range(len(XLearning)):
			XLearning[j][i] = XLearning[j][i] / MaxLearning[i]

	YLearning = YLearning.astype(float)

	MaxYLearning = 0
	for i in range(len(YLearning)):
		if(MaxYLearning < YLearning[i]):
			MaxYLearning = YLearning[i]
	# print ("MaxY = ", MaxY)
	for i in range(len(YLearning)):
		YLearning[i] = YLearning[i] / MaxYLearning

	# print (X)
	# print (Y)
	# create model


	# Fit the model
	print(XLearning.shape)
	for i in range(epochLoops):
		model.fit(XLearning[:trainLimitLearning], YLearning[:trainLimitLearning], epochs=epochsPerLoop, verbose=fittingVerbose, validation_split=0.2, batch_size=10)
		model.save('models/FullTrainingSet_epochs'+str((i+1)*epochsPerLoop)+'.h5')
		scores=model.evaluate(XLearning[:trainLimitLearning],YLearning[:trainLimitLearning],batch_size=10)
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


		# for j in  range(10):
		# 	rnd = trainLimit+numpy.random.randint(totalSize-trainLimit)
		# 	a = model.predict(numpy.array(X[rnd]).reshape(-1,6))
		# 	error_perc = abs(((Y[rnd]-a)*100)/Y[rnd])
		# 	print ("Y[",rnd,"] = ", Y[rnd], "a = ", a, " err_perc = ", error_perc)

		errorSumLearning=0
		totalReadingsLearning = 0
		for j in range((totalSizeLearning-1)):
			a = model.predict(numpy.array(XLearning[j]).reshape(-1,6))
			if(YLearning[j] == 0):
				continue
			error_percLearning = abs(((YLearning[j]-a)*100)/YLearning[j])
			errorSumLearning = errorSumLearning+error_percLearning
			totalReadingsLearning += 1
		print("\n\nAfter ", (i+1)*epochsPerLoop, "epochs,")
		print ("Avg Error in Learning Data Set = ", (errorSumLearning/(totalReadingsLearning)))
		# predictForTestSet(TestSetFilename)
		PredictForTestSetCollection(testSetFilenameCollection)

def LearnTestSet(filename):
	datasetLearning = numpy.loadtxt(filename, delimiter=",")

	totalSizeLearning = len(datasetLearning)-2
	# print(totalSize)
	trainLimitLearning = int(((len(datasetLearning)-2)/100)*100)
	# print(trainLimit)

	numpy.random.shuffle(datasetLearning)

	XLearning = numpy.zeros((totalSizeLearning,6))
	for j in range(totalSizeLearning):
		XLearning[j][0] = datasetLearning[j,0]
		XLearning[j][2] = datasetLearning[j,1]
		XLearning[j][4] = datasetLearning[j,2]
	for j in range(1,totalSizeLearning):
		XLearning[j][1] = XLearning[j-1][0]
		XLearning[j][3] = XLearning[j-1][2]
		XLearning[j][5] = XLearning[j-1][4]
	Y = datasetLearning[:,3]
	# print (X)
	# print (Y)

	MaxLearning = numpy.zeros(len(XLearning[0]))
	XLearning = XLearning.astype(float)
	for i in range(len(XLearning[0])):
		MaxLearning[i] = 0
		for j in range(len(XLearning)):
			if(MaxLearning[i] < XLearning[j][i]):
				MaxLearning[i] = XLearning[j][i]
		# print ("Max = ", Max[i])
		for j in range(len(XLearning)):
			XLearning[j][i] = XLearning[j][i] / MaxLearning[i]

	YLearning = YLearning.astype(float)

	MaxYLearning = 0
	for i in range(len(YLearning)):
		if(MaxYLearning < YLearning[i]):
			MaxYLearning = YLearning[i]
	# print ("MaxY = ", MaxY)
	for i in range(len(YLearning)):
		YLearning[i] = YLearning[i] / MaxYLearning

	# print (X)
	# print (Y)
	# create model


	# Fit the model
	print(XLearning.shape)
	for i in range(epochLoopsLearning):
		model.fit(XLearning[:trainLimitLearning], Y[:trainLimitLearning], epochs=epochsPerLoop, verbose=fittingVerbose, validation_split=0.2, batch_size=10)
		model.save('3_nn13_10_mean_squared_error_sgd_bs10_split02_epochs_'+str(i)+'0000.h5')
		scores=model.evaluate(XLearning[:trainLimitLearning],YLearning[:trainLimitLearning],batch_size=10)
		print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


		# for j in  range(10):
		# 	rnd = trainLimit+numpy.random.randint(totalSize-trainLimit)
		# 	a = model.predict(numpy.array(X[rnd]).reshape(-1,6))
		# 	error_perc = abs(((Y[rnd]-a)*100)/Y[rnd])
		# 	print ("Y[",rnd,"] = ", Y[rnd], "a = ", a, " err_perc = ", error_perc)

		errorSumLearning=0
		totalReadingsLearning = 0
		for j in range((totalSizLearninge-1)):
			a = model.predict(numpy.array(XLearning[j]).reshape(-1,6))
			if(YLearning[j] == 0):
				continue
			error_percLearning = abs(((YLearning[j]-a)*100)/YLearning[j])
			errorSumLearning = errorSumLearning+error_percLearning
			totalReadingLearnings += 1
		print("\n\nAfter ", (i+1)*epochsPerLoop, "epochs,")
		print ("Avg Error in Learning Data Set = ", (errorSumLearning/(totalReadingsLearning)))
		# predictForTestSet(TestSetFilename)
		PredictForTestSetCollection(testSetFilenameCollection)



# print (model)
# a = model.predict(numpy.array([0.00125628, 0, 0.07505039, 0.07033415, 0.06912259, 0.06125143]).reshape(-1,6))
# for i in  range(10):
# 	rnd = trainLimit+numpy.random.randint(totalSize-trainLimit)
# 	a = model.predict(numpy.array(X[rnd]).reshape(-1,6))
# 	error_perc = abs(((Y[rnd]-a)*100)/Y[rnd])
# 	print ("Y[",rnd,"] = ", Y[rnd], "a = ", a, " err_perc = ", error_perc)

# print (a)
# evaluate the model

learnFromLearningSetCollection(learningSetFilenameCollection)

# scores = model.evaluate(XLearning[:trainLimit], YLearning[:trainLimit])
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))