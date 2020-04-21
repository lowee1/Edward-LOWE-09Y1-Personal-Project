from os import path

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import *
from tkinter.font import Font
from tkinter.filedialog import askopenfilename
from tkinter.ttk import Progressbar

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

class page(Frame):

	def __init__(self, master,title,font=('Helvetica',60)):
		super(page,self).__init__(master)

		self.titleLabel = Label(self, text=title, font=font, relief=GROOVE)
		self.titleLabel.pack(pady=50,padx=30,ipadx=30,ipady=30)

		self.contentFrame = Frame(self)
		self.contentFrame.pack()

class scrollingListbox(Frame):

	def __init__(self,master,height=13,width=40):
		super(scrollingListbox,self).__init__(master)

		self.listbox = Listbox(self,selectmode=EXTENDED,height=height,width=width)
		self.listbox.grid(column=0,row=0)

		self.scrollbar = Scrollbar(self)
		self.scrollbar.grid(column=1,row=0,sticky=NS)

		self.listbox.config(yscrollcommand=self.scrollbar.set)
		self.scrollbar.config(command=self.listbox.yview)

	def insert(self,index,item):
		self.listbox.insert(index,item)

window = tk.Tk()
window.title('Machine Learning Creator')
window.state('zoomed')
window.config(bg='green')

problemType = StringVar(value="Text (Classification) -- Default")
continueVar = BooleanVar()
trainingDataDF = pd.DataFrame()

# homepage
HomePage = page(window,'Machine Learning Creator')
HomePage.pack(fill=BOTH,expand=1)
startButton = Button(HomePage.contentFrame,text='Start',command=lambda:continueVar.set(True),font=('helvetica',30))
startButton.grid(padx=20,pady=20)
startButton.wait_variable(continueVar)
HomePage.pack_forget()

# problem type selection
ProblemSelect = page(window,'Problem Type')
ProblemSelect.pack(fill=BOTH,expand=1)
explanationBox = Label(ProblemSelect.contentFrame)

problemTypeChoices = {'Images (Classification)': 'Predict a label from an image',
					  'Numbers (Regression)': 'Numerical data with continuous numerical output e.g. stock market data',
					  'Numbers (Classification)': 'Numerical data with fixed outputs e.g even and odd numbers',
					  'Text (Regression)': 'Text data with continuous numerical output e.g sentiment analysis',
					  'Text (Classification) -- Default': 'Text data with fixed outputs e.g spam filtering. Default option'}

for choice, description in problemTypeChoices.items():
	option = Radiobutton(ProblemSelect.contentFrame, text=choice, variable=problemType, value=choice,
							command=lambda description=description: explanationBox.config(text=description))
	option.grid(column=0, sticky=W, padx=5, pady=5)
map(lambda x: x.deselect(), list(ProblemSelect.contentFrame.children.values()))
list(ProblemSelect.contentFrame.children.values())[-1].invoke()

explanationBox.grid(column=1, row=3, sticky=E)
nxtBtn = Button(ProblemSelect.contentFrame, text="next",command=lambda:continueVar.set(True))
nxtBtn.grid(column=0, columnspan=2, padx=10, ipadx=30, ipady=5)

ProblemSelect.pack_forget()

# select which columns to use
DataCollecting = page(window,'Select Training Data')
DataCollecting.pack(fill=BOTH,expand=1)

# load data
filename = askopenfilename(initialdir='/', title='Choose Training Data', filetypes=[
							('CSV', '*.csv'), ('Excel spreadsheets', '*.xls *.xlsx *.xlsm *.xlsb')])
if path.splitext(filename)[1].lower() == '.csv':
	trainingDataDF = pd.read_csv(filename)
else:
	trainingDataDF = pd.read_excel(filename)

if 'Classification' in problemType.get():
	le = LabelEncoder()
	le.fit(np.array(list(set(trainingDataDF.to_numpy().flatten().tolist()))).reshape(-1,1).ravel())

# listbox with all the column names
columnListbox = scrollingListbox(DataCollecting.contentFrame,25)
columnListbox.grid(column=0,row=0,rowspan=9,padx=10,pady=10,sticky=NS)
for columName in trainingDataDF.columns:
	columnListbox.insert(END,columName)


featureAddButton = Button(DataCollecting.contentFrame,text='Add >>>')
featureAddButton.grid(column=1,row=1)

featureRemoveButton = Button(DataCollecting.contentFrame,text='<<< Remove')
featureRemoveButton.grid(column=1,row=2)

featureListbox = scrollingListbox(DataCollecting.contentFrame)
featureListbox.grid(column=2,row=0,rowspan=4,padx=10,pady=10)


targetAddButton = Button(DataCollecting.contentFrame,text='Add >>>')
targetAddButton.grid(column=1,row=5)

targetRemoveButton = Button(DataCollecting.contentFrame,text='<<< Remove')
targetRemoveButton.grid(column=1,row=6)

targetListbox = scrollingListbox(DataCollecting.contentFrame)
targetListbox.grid(column=2,row=4,rowspan=4,padx=10,pady=10)

window.mainloop()