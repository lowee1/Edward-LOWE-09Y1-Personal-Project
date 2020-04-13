import tkinter as tk
from os.path import splitext
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Progressbar

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, SVR


def chooseAlgorithm(problemType,features,targets):
	if problemType == 'classification':
		models = [('RFC', RandomForestClassifier()),
				  ('GNB', GaussianNB()),
				  ('KNC', KNeighborsClassifier()),
				  ('SVC', SVC()),
				  ('LDA', LinearDiscriminantAnalysis)]
	elif problemType == 'regression':
		models = [('RFR', RandomForestRegressor()),
				  ('LNR', LinearRegression()),
				  ('LGR', LogisticRegression())
				  ('KNR', KNeighborsClassifier),
				  ('SVR', SVR())]
	else:
		raise TypeError(['expected either \'classification\' or \'regression\' as problem type'])
	
	X = features.values
	Y = targets.values

	results = []
	names = []
	scoring = 'accuracy'
	for name, model in models:
		kfold = KFold(n_splits=5, random_state=7,shuffle=True)
		cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		print(msg)

class MachineLearningCreator(tk.Tk):

	def __init__(self):

		tk.Tk.__init__(self)
		self.container = Frame(self, relief=GROOVE)

		self.container.grid(column=0, row=0, padx=10, pady=10)

		self.container.grid_rowconfigure(0, weight=1)
		self.container.grid_columnconfigure(0, weight=1)

		self.problemType = StringVar(
			value="   Text (Classification) -- Default")

		self.frames = {HomePage: HomePage(self.container, self)}

		self.frames[HomePage].grid()

	def collectData(self):
		self.frames[HomePage].destroy()
		self.frames = {DataCollecting: DataCollecting(self.container, self)}
		self.frames[DataCollecting].tkraise()
		self.frames[DataCollecting].grid()
		self.frames[DataCollecting].getData()
		self.frames[DataCollecting].chooseColumns()

	def createModel(self):
		self.frames[DataCollecting].pack_forget()
		self.frames[DataCollecting].columnChoiceFrame.pack_forget()
		self.frames = {Creating: Creating(self.container, self)}
		self.frames[Creating].tkraise()
		self.frames[Creating].grid()

class HomePage(Frame):

	def __init__(self, parent, controller):

		Frame.__init__(self, parent)

		titleLabel = Label(self, text='   Machine Learning Creator', font=(
			'Verdana', 30), relief=GROOVE)
		titleLabel.grid(column=0, row=0, padx=20, pady=30, ipadx=40, ipady=30)

		radioFrame = Frame(self, relief=SUNKEN)
		radioFrame.grid(column=0, row=1, padx=10, pady=10, ipadx=30, ipady=30)

		explanationBox = Label(radioFrame)

		problemTypeChoices = {'   Images (Classification)': 'Predict a label from an image',
							  '   Numbers (Regression)': 'Numerical data with continuous numerical output e.g. stock market data',
							  '   Numbers (Classification)': 'Numerical data with fixed outputs e.g even and odd numbers',
							  '   Text (Regression)': 'Text data with continuous numerical output e.g sentiment analysis',
							  '   Text (Classification) -- Default': 'Text data with fixed outputs e.g spam filtering. Default option'}

		for choice, description in problemTypeChoices.items():
			option = Radiobutton(radioFrame, text=choice, variable=controller.problemType, value=choice,
								 command=lambda description=description: explanationBox.config(text=description))
			option.grid(column=0, sticky=W, padx=5, pady=5)
		map(lambda x: x.deselect(), list(radioFrame.children.values()))
		list(radioFrame.children.values())[-1].invoke()

		explanationBox.grid(column=1, row=3, sticky=E)

		nxtBtn = Button(radioFrame, text="next",
						command=controller.collectData)
		nxtBtn.grid(column=0, columnspan=2, padx=10, ipadx=30, ipady=5)

class DataCollecting(Frame):

	def __init__(self, parent, controller):
		Frame.__init__(self, parent)
		titleLabel = Label(self, text='Selecting Training Data',
						   font=('Verdana', 25), relief=GROOVE)
		titleLabel.grid(column=0, row=0, padx=20, pady=10, ipadx=100, ipady=30)
		self.controller = controller

		self.columnChoiceFrame = Frame(self, relief=SUNKEN)
		self.columnChoiceFrame.grid(
			column=0, row=1, padx=20, pady=10, ipadx=50, ipady=30)

	def getData(self):
		filename = askopenfilename(initialdir='/', title='Choose Training Data', filetypes=[
								   ('CSV', '*.csv'), ('Excel spreadsheets', '*.xls *.xlsx *.xlsm *.xlsb')])
		if splitext(filename)[1].lower() == '.csv':
			self.trainingDataDF = pd.read_csv(filename)
		else:
			self.trainingDataDf = pd.read_excel(filename)

		if 'Classification' in self.controller.problemType.get():
			self.le = LabelEncoder()
			self.le.fit(np.array(list(set(self.trainingDataDF.to_numpy().flatten().tolist()))).reshape(-1,1))
		
	def chooseColumns(self):
		self.featureChoices = []
		self.targetChoices = []
		featureLabel = Label(self.columnChoiceFrame,
							 text='Feature', font=('Verdana', 18))
		featureLabel.grid(column=1, row=0)
		targetLabel = Label(self.columnChoiceFrame,
							text='Target', font=('Verdana', 18))
		targetLabel.grid(column=3, row=0)

		for i in range(len(self.trainingDataDF.columns)):
			feature = IntVar()
			target = IntVar()
			columnName = self.trainingDataDF.columns[i]
			self.featureChoices.append(feature)
			self.targetChoices.append(target)
			columnLabel = Label(self.columnChoiceFrame,
								text=columnName, font=('Verdana', 12))
			columnLabel.grid(column=0, padx=10, pady=10)
			featureButton = Checkbutton(self.columnChoiceFrame, indicatoron=0,
										width=10, variable=feature, font=('Verdana', 12), relief=SUNKEN)
			featureButton.grid(column=1, row=i+1, padx=10)
			featureButton.deselect()
			targetButton = Checkbutton(self.columnChoiceFrame, indicatoron=0,
									   width=10, variable=target, font=('Verdana', 12), relief=SUNKEN)
			targetButton.grid(column=3, row=i+1, padx=10)
			targetButton.deselect()

		self.featureChoices[0].set(1)
		self.targetChoices[-1].set(1)

		startButton = Button(self.columnChoiceFrame, text='Next', command=lambda: [
							 self.assignData(), self.controllercreateModel()])
		startButton.grid(column=2)

	def assignData(self):
		self.featureIndexes = [index for index, value in enumerate(list(map(lambda value:value.get(), self.featureChoices))) if value == 1]
		self.featureColumnNames = [list(self.trainingDataDF.columns)[i] for i in self.featureIndexes]
		self.featureTrain = self.trainingDataDF[self.featureColumnNames].copy()

		self.targetIndexes = [index for index, value in enumerate(list(map(lambda value:value.get(), self.targetChoices))) if value == 1]
		self.targetColumnNames = [list(self.trainingDataDF.columns)[i] for i in self.targetIndexes]
		self.targetTrain = self.trainingDataDF[self.targetColumnNames].copy()

class Creating(Frame):

	def __init__(self, parent, controller):

		Frame.__init__(self, parent)

		titleLabel = Label(
			self, text='Creating your ml model', font=('Verdana', 12))
		titleLabel.grid(column=0, row=0)

		progress = Progressbar(self, orient=HORIZONTAL, length=400)
		progress.grid(column=0, row=1)
		progress.config(mode='indeterminate')
		progress.start()

		Data = parent.children['!datacollecting']

		if 'Images' in controller.problemType.get():
			# not done yet
			print()
		elif 'Classification' in controller.problemType.get():
			chooseAlgorithm('classification',Data.featureTrain,Data.targetTrain)
		else:
			chooseAlgorithm('regression',Data.featureTrain,Data.targetTrain)


app = MachineLearningCreator()
app.minsize(700, 400)
app.title('Machine Learning Creator')

app.mainloop()
