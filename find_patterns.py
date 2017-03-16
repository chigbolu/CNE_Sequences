import csv as csv
import math
import subprocess
import numpy as np
import pandas as pd
from collections import defaultdict


raw_data = pd.read_csv('matches.csv', header=0)

#data = [value for value in raw_data]
#data = np.array(data)

#Extract columns "CNE Name"and "Symbol"  
data_CNE_SYM = raw_data[['CNE_NAME','SYMBOL']]

seq_dict = defaultdict(list)

# insert CNEs in the dictionary with the Symbols bound
for index, row in data_CNE_SYM.iterrows():
	thisCNE = row["CNE_NAME"]
	thisSYM = row["SYMBOL"]
	cneToTrack = []
#store only the first name of the sequence that are separated by semi-colums
	if ":" not in thisCNE : 
		seq_dict[thisCNE].append(thisSYM)
	else:
		cneToTrack = thisCNE.split(":")
		seq_dict[cneToTrack[0]].append(thisSYM)

# print a CNE sample to see if the listing is correct
#print(seq_dict["CRCNE00000134"])

#store symbol or numbers representing molecules in the database
sym_dict = defaultdict(list)
symbols = []

for index, row in data_CNE_SYM.iterrows():
	thisSYM = row["SYMBOL"]
	sym_dict[thisSYM]

#this could be replaced with a normal dictionary that does not have array [] 
for key in sym_dict:
	symbols.append(key)

#store unique CNE name in dataframe
cne_names = []
for key in seq_dict:
	cne_names.append(key)


#sort list of symbols: letters before numbers
symbols = sorted(symbols, key=lambda x: (x[0].isdigit(), x))

#create new dataframe with a column for each symbol 

#extract column CNE_NAME
cne_list = pd.DataFrame(cne_names)
#rename name of the column 
cne_list.rename(columns={0:'CNE_NAME'}, inplace=True)

#create a column for each symbol

i = 1
for s in symbols:
	cne_list.insert(i, s, 0)
	i+=1

#print cne_list

#populate the dataframe with the data from the database - how many times is each symbol bound?

for index,row in cne_list.iterrows():

	 #row[seq_dict[row['CNE_NAME'][0]]] += 1
	for symb in seq_dict[row['CNE_NAME']]:
		cne_list.ix[index,symb] += 1

for index,row in cne_list.iterrows():
	print row







	







			
	
	








