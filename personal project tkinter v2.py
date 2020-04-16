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

class page(Frame):

	def __init__(self, master=None):
		super().__init__(master=master)
		Frame.__init__(self,master)

		self.title = 'Title'

		titleLabel = Label(self, text=self.title, font=(
			'Verdana', 30), relief=GROOVE)
		titleLabel.grid(column=0, row=0, padx=20, pady=30, ipadx=40, ipady=30)

		contentFrame = Frame(self)