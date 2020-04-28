from os import path
from joblib import dump,load
from datetime import datetime
from time import sleep

from numpy import array
from pandas import DataFrame,read_csv,read_excel

import tkinter as tk
from tkinter import Tk,Frame,Label,Button,Radiobutton,Listbox,Scrollbar,StringVar,BooleanVar
from tkinter.font import Font
from tkinter.filedialog import askopenfilename,asksaveasfilename
from tkinter.ttk import Progressbar,Separator

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression,SGDRegressor,SGDClassifier,Lasso,ElasticNet,Ridge
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR,LinearSVC,LinearSVR



# Function to score multiple algorithms
# Author: Jason Brownlee
# Date: 13 Dec 2019
# Availability: https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

def chooseAlgorithm(problemType,features,targets):
	if 'Classification' in problemType:
		models = {'RFC': RandomForestClassifier(),
				  'ETC': ExtraTreesClassifier,
				  'GNB': GaussianNB(),
				  'MNB': MultinomialNB(),
				  'KNC': KNeighborsClassifier(),
				  'SVC': SVC(),
				  'LSVC': LinearSVC(),
				  'LGR': LogisticRegression(),
				  'LDA': LinearDiscriminantAnalysis(),
				  'SDGC': SGDClassifier}
	elif 'Regression' in problemType:
		models = {'RFR': RandomForestRegressor(),
				  'ETR': ExtraTreesClassifier(),
				  'LNR': LinearRegression(),
				  'SDGR': SGDRegressor,
				  'KNR': KNeighborsClassifier(),
				  'SVR': SVR(),
				  'LSVR': LinearSVR(),
				  'Lasso':Lasso(),
				  'ENET':ElasticNet(),
				  'Ridge':Ridge()}
	else:
		raise TypeError(['expected either \'classification\' or \'regression\' as problem type'])

	results = {}
	X_train, X_test, y_train, y_test = train_test_split(features, targets.values.ravel())
	for name, model in models.items():
		model.fit(X_train,y_train)
		score = model.score(X_test,y_test)
		results[name] = score

	bestModelScore = sorted(results.items(),key=lambda x: x[1],reverse=True)[0]

	model = models[bestModelScore[0]].fit(features,targets)

	return model

def addToListBox(fromListbox,toListbox):
	selection = [fromListbox.listbox.get(i) for i in fromListbox.listbox.curselection()]
	for item in selection:
		toListbox.listbox.insert('end',item)

def deleteFromListBox(fromListbox):
	selection = fromListbox.listbox.curselection()
	for i in reversed(selection):
		fromListbox.listbox.delete(i)

class page(Frame):

	def __init__(self, master,title,font=('Helvetica',60)):
		super(page,self).__init__(master)

		self.titleLabel = Label(self, text=title, font=font, relief='groove')
		self.titleLabel.pack(pady=40,padx=30,ipadx=30,ipady=30)

		self.contentFrame = Frame(self)
		self.contentFrame.pack()

class scrollingListbox(Frame):

	def __init__(self,master,height=10,width=45):
		super(scrollingListbox,self).__init__(master)

		self.listbox = Listbox(self,selectmode='extended',height=height,width=width)
		self.listbox.grid(column=0,row=0)

		self.scrollbar = Scrollbar(self)
		self.scrollbar.grid(column=1,row=0,sticky='ns')

		self.listbox.config(yscrollcommand=self.scrollbar.set)
		self.scrollbar.config(command=self.listbox.yview)

def homePage(window):
	HomePage = page(window,'Machine Learning Creator')
	HomePage.pack()

	makeOrUse = StringVar()

	startButton = Button(HomePage.contentFrame,text='Start',command=lambda:makeOrUse.set('make'),font=('helvetica',30))
	startButton.pack(pady=20)

	predictButton = Button(HomePage.contentFrame,text='Predict',command=lambda:makeOrUse.set('use'),font=('helvetica',30))
	predictButton.pack(pady=20)

	HomePage.wait_variable(makeOrUse)
	HomePage.pack_forget()

	if makeOrUse.get() == 'make':
		makeModel(window)
	else:
		useModel(window)

def makeModel(window):
	ProblemSelect = page(window,'Problem Type')
	ProblemSelect.pack()

	problemType = StringVar(value="Text (Classification) -- Default")
	continueVar = BooleanVar()

	explanationBox = Label(ProblemSelect.contentFrame)

	problemTypeChoices = {'Images': 'Predict a label from an image',
						'Numbers (Regression)': 'Numerical data with continuous numerical output e.g. stock market data',
						'Numbers (Classification)': 'Numerical data with fixed outputs e.g even and odd numbers',
						'Text (Regression)': 'Text data with continuous numerical output e.g sentiment analysis',
						'Text (Classification) -- Default': 'Text data with fixed outputs e.g spam filtering. Default option'}

	for choice, description in problemTypeChoices.items():
		option = Radiobutton(ProblemSelect.contentFrame, text=choice, variable=problemType, value=choice,
								command=lambda description=description: explanationBox.config(text=description))
		option.grid(column=0, sticky='w', padx=5, pady=5)
	map(lambda x: x.deselect(), list(ProblemSelect.contentFrame.children.values()))
	list(ProblemSelect.contentFrame.children.values())[-1].invoke()

	explanationBox.grid(column=1, row=3, sticky='e')
	nxtBtn = Button(ProblemSelect.contentFrame, text="next",command=lambda:continueVar.set(True))
	nxtBtn.grid(column=1, columnspan=2, padx=10, ipadx=30, ipady=5)

	ProblemSelect.wait_variable(continueVar)
	ProblemSelect.pack_forget()

	# select which columns to use
	DataCollecting = page(window,'Select Training Data')
	DataCollecting.pack()

	# load data
	filename = askopenfilename(title='Choose Training Data', filetypes=[
								('CSV', '*.csv'), ('Excel spreadsheets', '*.xls *.xlsx *.xlsm *.xlsb')])
	if path.splitext(filename)[1].lower() == '.csv':
		trainingDataDF = read_csv(filename)
	else:
		trainingDataDF = read_excel(filename)

	# If you didn't clean your data, I'm just going to pad it. Serves you right.
	trainingDataDF = trainingDataDF.apply(lambda x: x.interpolate(method='pad'))

	# listbox with all the column names
	columnListbox = scrollingListbox(DataCollecting.contentFrame,20)
	columnListbox.grid(column=0,row=0,rowspan=8,padx=10,pady=10,sticky='ns')
	for columName in trainingDataDF.columns:
		columnListbox.listbox.insert('end',columName)


	featureListbox = scrollingListbox(DataCollecting.contentFrame)
	featureListbox.grid(column=2,row=0,rowspan=4,padx=10,pady=10)

	featureAddButton = Button(DataCollecting.contentFrame,text='Add >>>',
							command=lambda:addToListBox(columnListbox,featureListbox))
	featureAddButton.grid(column=1,row=1)

	featureRemoveButton = Button(DataCollecting.contentFrame,text='<<< Remove',
								command=lambda:deleteFromListBox(featureListbox))
	featureRemoveButton.grid(column=1,row=2)


	targetListbox = scrollingListbox(DataCollecting.contentFrame)
	targetListbox.grid(column=2,row=4,rowspan=4,padx=10,pady=10)

	targetAddButton = Button(DataCollecting.contentFrame,text='Add >>>',
							command=lambda:addToListBox(columnListbox,targetListbox))
	targetAddButton.grid(column=1,row=5)

	targetRemoveButton = Button(DataCollecting.contentFrame,text='<<< Remove',
								command=lambda:deleteFromListBox(targetListbox))
	targetRemoveButton.grid(column=1,row=6)

	collectDataButton = Button(DataCollecting.contentFrame,text='Create',command=lambda:continueVar.set(True))
	collectDataButton.grid(column=2,row=8,pady=20,ipadx=20)

	DataCollecting.wait_variable(continueVar)
	DataCollecting.pack_forget()

	creating = page(window,'Creating')
	creating.pack()

	progress = Progressbar(creating.contentFrame)
	progress.pack()
	progress.config(mode='indeterminate')
	progress.start()

	sleep(2)

	featureColumnNames = featureListbox.listbox.get(0,'end')
	targetColumnNames = targetListbox.listbox.get(0,'end')

	featureTrain = trainingDataDF[list(featureColumnNames)]
	targetTrain = trainingDataDF[list(targetColumnNames)]

	if 'Text' in problemType.get():
		featureEncoder = LabelEncoder()
		featureEncoder.fit(array(list(set(featureTrain.to_numpy().flatten().tolist()))).reshape(-1,1).ravel())
		featureTrain = featureTrain.applymap(lambda x: featureEncoder.transform(array(x).reshape(1,1))[0])

	if 'Classification' in problemType.get():
		targetEncoder = LabelEncoder()
		targetEncoder.fit(array(list(set(targetTrain.to_numpy().flatten().tolist()))).reshape(-1,1).ravel())
		targetTrain = targetTrain.applymap(lambda x: targetEncoder.transform(array(x).reshape(1,1))[0])

	model = chooseAlgorithm(problemType.get(),featureTrain,targetTrain)

	progress.stop()
	progress.pack_forget()

	modelname = str(model.__class__).split('.')[-1][:-2]
	filename = modelname + ' ' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
	filepath = asksaveasfilename(initialfile=filename,defaultextension='.mlmc',
								 filetypes=[('Edward Machine Learning Creater Model','*.emlcm')],
								 title='Save As')
	dump(model,filepath,5)

	backButton = Button(window,text='Home Page',font=('Helvetica',30),
						command=lambda:[continueVar.set(True),homePage(window)])
	backButton.pack(pady=20)

	quitButton = Button(window,text='Quit',font=('Helvetica',30),
						command=lambda:[continueVar.set(True),window.destroy()])
	quitButton.pack(pady=20)

	creating.wait_variable(continueVar)
	creating.destroy()

def useModel(window):
	PredictPage = page(window,'Predict')
	PredictPage.grid()

	filename = askopenfilename(filetypes=[('Edward Machine Learning Creator Model','*.emlcm')],title='Load Model')
	model = load(filename)

	manualInputButton = Button(PredictPage.contentFrame,text='manually input features')
	manualInputButton.grid(column=0,row=0)

	inputSeparator = Separator(PredictPage.contentFrame,orient='vertical')
	inputSeparator.grid(column=1,row=0)

	loadFeaturesButton = Button(PredictPage.contentFrame,text='load features')
	loadFeaturesButton.grid(column=2,row=0)


window = tk.Tk()
window.title('Machine Learning Creator')
window.state('zoomed')

homePage(window)

window.mainloop()