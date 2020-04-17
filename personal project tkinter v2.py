import tkinter as tk
from os import path
from tkinter import *
from tkinter.font import Font
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

problemType = StringVar(value="Text (Classification) -- Default")

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

	def __init__(self, master):
		super().__init__(master)
		Frame.__init__(self,master)

		self.title = 'Title'
		self.font = Font()

		titleLabel = Label(self, text=self.title, font=(
			'Verdana', 30), relief=GROOVE)
		titleLabel.grid(column=0, row=0, padx=20, pady=30, ipadx=40, ipady=30)

		self.contentFrame = Frame(self)
		self.contentFrame.grid(column=0, row=1, padx=10, pady=10, ipadx=30, ipady=30)

window = tk.Tk()

HomePage = page(window)
HomePage.title = 'Machine Learning Creator'
startButton = Button(HomePage.contentFrame,text='Start')


window.title('Machine Learning Creator')
window.mainloop()