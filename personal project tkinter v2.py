import sys

from os import path
from math import sqrt
from time import sleep
from joblib import dump,load
from datetime import datetime

from numpy import array
from pandas import DataFrame,read_csv,read_excel,concat

import tkinter as tk
from tkinter.font import Font
from tkinter.ttk import Progressbar,Separator
from tkinter.filedialog import askopenfilename,asksaveasfilename
from tkinter import Tk,Frame,Label,Button,Radiobutton,Listbox,Scrollbar,StringVar,BooleanVar,messagebox

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR,LinearSVC,LinearSVR
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression,SGDRegressor,SGDClassifier,Lasso,ElasticNet,Ridge

retryError = 'Too many retries. Going to exit'
dataFiletypes = [('CSV', '*.csv'), ('Excel spreadsheets', '*.xls *.xlsx *.xlsm *.xlsb')]

# Function to score multiple algorithms
# Author: Jason Brownlee
# Date: 13 Dec 2019
# Availability: https://machinelearningmastery.com/compare-machine-learning-algorithms-python-scikit-learn/

def chooseAlgorithm(problemType,features,targets):
	if 'Classification' in problemType:
		models = {'RFC': RandomForestClassifier(),
				  'ETC': ExtraTreesClassifier(),
				  'GNB': GaussianNB(),
				  'MNB': MultinomialNB(),
				  'KNC': KNeighborsClassifier(n_neighbors=round(sqrt(len(features.index)))),
				  'SVC': SVC(),
				  'LSVC': LinearSVC(),
				  'LGR': LogisticRegression(),
				  'LDA': LinearDiscriminantAnalysis(),
				  'SDGC': SGDClassifier()}
	elif 'Regression' in problemType:
		models = {'RFR': RandomForestRegressor(),
				  'ETR': ExtraTreesRegressor(),
				  'LNR': LinearRegression(),
				  'SDGR': SGDRegressor(),
				  'KNR': KNeighborsRegressor(n_neighbors=round(sqrt(len(features.index)))),
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

class ScrollingListbox(Frame):

	def __init__(self,master,height=10,width=45):
		super(ScrollingListbox,self).__init__(master)

		self.listbox = Listbox(self,selectmode='extended',height=height,width=width)
		self.listbox.grid(column=0,row=0)

		self.scrollbar = Scrollbar(self)
		self.scrollbar.grid(column=1,row=0,sticky='ns')

		self.listbox.config(yscrollcommand=self.scrollbar.set)
		self.scrollbar.config(command=self.listbox.yview)

def HomePage(window):
	Home_Page = page(window,'Machine Learning Creator')
	Home_Page.pack()

	makeOrUse = StringVar()

	startButton = Button(Home_Page.contentFrame,text='Start',command=lambda:makeOrUse.set('make'),font=('helvetica',30))
	startButton.pack(pady=20)

	predictButton = Button(Home_Page.contentFrame,text='Predict',command=lambda:makeOrUse.set('use'),font=('helvetica',30))
	predictButton.pack(pady=20)

	Home_Page.wait_variable(makeOrUse)
	Home_Page.pack_forget()

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

	problemTypeChoices = {'Numbers (Regression)': 'Numerical data with continuous numerical output e.g. stock market data',
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
	fileNotLoaded = True
	counter = 0
	while fileNotLoaded:
		if counter == 10:
			messagebox.showerror(title='Error',message=retryError)
			sys.exit()
		try:
			counter += 1
			filename = askopenfilename(title='Choose Training Data', filetypes=dataFiletypes)
			if path.splitext(filename)[1].lower() == '.csv':
				trainingDataDF = read_csv(filename)
			else:
				trainingDataDF = read_excel(filename)
			# If you didn't clean your data, I'm just going to destroy it. Serves you right.
			trainingDataDF = trainingDataDF.apply(lambda x: x.interpolate(method='pad'))
			trainingDataDF = trainingDataDF.dropna(how='any')
			if len(trainingDataDF.index) < 50:
				raise ValueError(': Not enough data. Have to have at least 50 samples of data.\n'
								 'If you think you have enough, it might be because there were'
								 'invalid values that were automatically taken out.')
		except Exception as e:
			messagebox.showerror(title='Error',message=str(type(e)).split('\'')[1]+str(e))
			continue
		fileNotLoaded = False

	# listbox with all the column names
	columnListbox = ScrollingListbox(DataCollecting.contentFrame,20)
	columnListbox.grid(column=0,row=0,rowspan=8,padx=10,pady=10,sticky='ns')
	for columName in trainingDataDF.columns:
		columnListbox.listbox.insert('end',columName)

	featureListbox = ScrollingListbox(DataCollecting.contentFrame)
	featureListbox.grid(column=2,row=0,rowspan=4,padx=10,pady=10)

	featureAddButton = Button(DataCollecting.contentFrame,text='Add >>>',
							command=lambda:addToListBox(columnListbox,featureListbox))
	featureAddButton.grid(column=1,row=1)

	featureRemoveButton = Button(DataCollecting.contentFrame,text='<<< Remove',
								command=lambda:deleteFromListBox(featureListbox))
	featureRemoveButton.grid(column=1,row=2)


	targetListbox = ScrollingListbox(DataCollecting.contentFrame)
	targetListbox.grid(column=2,row=4,rowspan=4,padx=10,pady=10)

	targetAddButton = Button(DataCollecting.contentFrame,text='Add >>>',
							command=lambda:addToListBox(columnListbox,targetListbox))
	targetAddButton.grid(column=1,row=5)

	targetRemoveButton = Button(DataCollecting.contentFrame,text='<<< Remove',
								command=lambda:deleteFromListBox(targetListbox))
	targetRemoveButton.grid(column=1,row=6)

	collectDataButton = Button(DataCollecting.contentFrame,text='Create',
							   command=lambda:continueVar.set(True)
							   if len(featureListbox.listbox.get(0,'end')) > 0 and len(targetListbox.listbox.get(0,'end')) > 0
							   else messagebox.showwarning(title='Warning',message='You must have at least one feature and one target'))
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

	featureEncoder = LabelEncoder()
	targetEncoder = LabelEncoder()

	if 'Text' in problemType.get():
		featureEncoder.fit(array(list(set(featureTrain.to_numpy().flatten().tolist()))).reshape(-1,1).ravel())
		featureTrain = featureTrain.applymap(lambda x: featureEncoder.transform(array(x).reshape(1,1))[0])

	if 'Classification' in problemType.get():
		targetEncoder.fit(array(list(set(targetTrain.to_numpy().flatten().tolist()))).reshape(-1,1).ravel())
		targetTrain = targetTrain.applymap(lambda x: targetEncoder.transform(array(x).reshape(1,1))[0])

	model = chooseAlgorithm(problemType.get(),featureTrain,targetTrain)

	progress.stop()
	progress.pack_forget()

	modelname = str(model.__class__).split('.')[-1][:-2]
	filename = modelname + ' ' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

	fileNotLoaded = True
	counter = 0
	while fileNotLoaded:
		if counter == 10:
			messagebox.showerror(title='Error',message=retryError)
			sys.exit()
		try:
			counter += 1
			filepath = asksaveasfilename(initialfile=filename,defaultextension='.mlmc',
							filetypes=[('Edward Machine Learning Creater Model','*.emlcm')],
							title='Save As')
			dump([model,problemType.get(),featureEncoder,targetEncoder,featureTrain.columns,targetTrain.columns],filepath,5)
		except Exception as e:
			messagebox.showerror(title='Error',message=str(type(e)).split('\'')[1]+str(e))
			continue
		fileNotLoaded = False

	backButton = Button(creating.contentFrame,text='Back to Home Page',font=('Helvetica',30),
						command=lambda:[continueVar.set(True),creating.destroy(),HomePage(window)])
	backButton.pack(pady=20)

	quitButton = Button(creating.contentFrame,text='Quit',font=('Helvetica',30),
						command=lambda:[continueVar.set(True),window.destroy()])
	quitButton.pack(pady=20)

	creating.wait_variable(continueVar)

def useModel(window):
	PredictPage = page(window,'Predict')
	PredictPage.pack()

	fileNotLoaded = True
	counter = 0
	while fileNotLoaded:
		if counter == 10:
			messagebox.showerror(title='Error',message=retryError)
			sys.exit()
		try:
			counter += 1
			filename = askopenfilename(filetypes=[('Edward Machine Learning Creator Model','*.emlcm')],title='Load Model')
			if len(load(filename)) != 6:
				raise ValueError('File is invalid')
			model,problemType,featureEncoder,targetEncoder,featureColumns,targetColumn = load(filename)
			if not(
				   ('sklearn' in str(type(model))) and
				   ('Classification' in problemType or 'Regression' in problemType) and
				   ('Text' in problemType or 'Regression' in problemType) and
				   ('LabelEncoder' in str(type(featureEncoder))) and
				   ('LabelEncoder' in str(type(targetEncoder))) and
				   ('pandas.core.indexes.base.Index' in str(type(featureColumns)))and
				   ('pandas.core.indexes.base.Index' in str(type(targetColumn)))
				   ):
				raise ValueError('File is invalid')
		except Exception as e:
			messagebox.showerror(title='Error',message=str(type(e)).split('\'')[1]+str(e))
			continue
		fileNotLoaded = False

	sleep(1)

	fileNotLoaded = True
	counter = 0
	while fileNotLoaded:
		if counter == 10:
			messagebox.showerror(title='Error',message=retryError)
			sys.exit()
		try:
			counter += 1
			filename = askopenfilename(filetypes=dataFiletypes,
									   title='Load Features')
			if path.splitext(filename)[1].lower() == '.csv':
				features = read_csv(filename)
			else:
				features = read_excel(filename)
			features = features.dropna(how='any')
			if features.columns.tolist() != featureColumns.tolist():
				raise ValueError(' incorrect features (columns)')
		except Exception as e:
			messagebox.showerror(title='Error',message=str(type(e)).split('\'')[1]+str(e))
			continue
		fileNotLoaded = False

	results = DataFrame(model.predict(features),columns=targetColumn)
	results = concat([features,results],axis=1)

	messagebox.showinfo(title='Save',message='Finished prediction. Saving results to file now')

	fileNotLoaded = True
	counter = 0
	while fileNotLoaded:
		if counter == 10:
			messagebox.showerror(title='Error',message=retryError)
			sys.exit()
		try:
			counter += 1
			filename = asksaveasfilename(filetypes=dataFiletypes,
										 title='Save Results',
										 initialfile=path.split(path.splitext(filename)[0])[1]+
										 			 'results'+datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
										 defaultextension='.csv'
													  )
			if path.splitext(filename)[1].lower() == '.csv':
				results.to_csv(filename,index=False)
			else:
				results.to_excel(filename,index=False)
		except Exception as e:
			messagebox.showerror(title='Error',message=str(type(e)).split('\'')[1]+str(e))
			continue
		fileNotLoaded = False

	continueVar = BooleanVar()
	backButton = Button(PredictPage.contentFrame,text='Home Page',font=('Helvetica',30),
						command=lambda:[continueVar.set(True),PredictPage.destroy(),HomePage(window)])
	backButton.pack(pady=20)

	quitButton = Button(PredictPage.contentFrame,text='Quit',font=('Helvetica',30),
						command=lambda:[continueVar.set(True),window.destroy()])
	quitButton.pack(pady=20)

	PredictPage.wait_variable(continueVar)

window = tk.Tk()
window.title('Machine Learning Creator')
window.state('zoomed')

HomePage(window)

window.mainloop()