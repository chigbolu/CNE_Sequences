import csv as csv
import math
import subprocess
import numpy as np
import pandas as pd
import random
import subprocess
from collections import defaultdict
from scipy.stats import entropy
from sklearn.metrics.cluster import adjusted_rand_score


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

############create new dataframe with a column for each symbol #############

#extract column CNE_NAME
cne_list = pd.DataFrame(cne_names)
#rename name of the column 
cne_list.rename(columns={0:'CNE_NAME'}, inplace=True)

#create a column for each symbol
i = 1
for s in symbols:
	cne_list.insert(i, s, 0.01)
	i+=1

#print cne_list

#populate the dataframe with the data from the database - how many times is each symbol bound?

for index,row in cne_list.iterrows():

	 #row[seq_dict[row['CNE_NAME'][0]]] += 1
	for symb in seq_dict[row['CNE_NAME']]:
		cne_list.ix[index,symb] += 1

#for index,row in cne_list.iterrows():
#	print row

############# Normalise data to probability distribution values. Each rows sums 1 ####################


#create a new column sumRow that will be used for normalisation
cne_list['sumRow'] = cne_list.sum(axis=1) 


#normalise columns to probability distribution values
for index,row in cne_list.iterrows():
	for key in sym_dict:
		norm_value = float(row[key])/row['sumRow']
		cne_list.ix[index,key] = norm_value
		
	
#get rid of sumRow column  and CNE_NAME column
del cne_list['sumRow']

cne_names = cne_list['CNE_NAME']
del cne_list['CNE_NAME']

 
########################################################Application of Kmeans with KL and Euclidean############################################################

#convert dataframe to list as the code I had developed for running kmeans uses lists

cne_data = cne_list.values.tolist()
#convert cne_names to list
cne_names = cne_names.values.tolist()

#i = 0
#for row in cne_names:
#	if(i< 10):
#		print row
#	i += 1


#************************************************* MAIN METHOD *****************************************************#

def main():


	cnes_objs = []
	for row in cne_data:
		cnes_objs.append(CNE(row))

# Calculate clusters applying Kmeans with Euclidean distance and Kmeans with KL distance

	clustersE = kmeans(cnes_objs, 3, get_E_Distance)
	clustersK = kmeans(cnes_objs, 3, get_K_Distance)


# Obtain cluster values (0,1,2) in order to measure clustering similarity
	
	clusterValuesE = []
	clusterValuesK = []
	i = 0


	while i < 3:
		
		CNEsE = clustersE[i].CNEs
		CNEsK = clustersK[i].CNEs

		for f in CNEsE:
			clusterValuesE.append(i)
		for f in CNEsK:
			clusterValuesK.append(i)
		i += 1

	
	for row in CNEsE:
		print row

	for row in CNEsK:
		print row

	print "The Rand Index between the two clustering algorithms is:", adjusted_rand_score(clusterValuesK,clusterValuesE)





#******************************** CNE CLASS ****************************************#


class CNE:
	#a CNE might bind 30 different types of molecules
	def __init__(self,features):
	
		self.features = features
		self.n = len(features)



	def __repr__(self):
 	       return str(self.features)


#******************************* CLUSTER CLASS ***************************************#

class Cluster:
	def __init__(self,CNEs):
		if len(CNEs) == 0: raise Exception("ILLEGAL: empty cluster")
	
		#the features that belong to this cluster
		self.CNEs = CNEs
		self.n = CNEs[0].n
		self.centroid = self.calculateCentroid()

		 # The number of featues
		

	def __repr__(self):
       
     #   String representation of this object
       
		return str(self.CNEs)

	def update(self,CNEs,getDistance):

		#returns the distance between previous and the new centroid after recalculating and storing the new centroid

		old_centroid = self.centroid
		self.CNEs = CNEs
		self.centroid = self.calculateCentroid()
		
	        shift = getDistance(old_centroid,self.centroid)
		return shift

	def calculateCentroid(self):
	
		numCNEs = len(self.CNEs)
		features = [f.features for f in self.CNEs]
		# Reformat that so all x's are together, all y'z etc.
		unzipped = zip(*features)
		# Calculate the mean for each dimension

		centroid_features = [math.fsum(dList)/numCNEs for dList in unzipped]

		return CNE(centroid_features)


#***************************** KMEANS FUNCTION ****************************************#


def kmeans(CNEs, k, getDistance):

         
	initial = random.sample(CNEs, k)
	clusters = [Cluster([f]) for f in initial]

	loopCounter = 0
	
	while True:
	
		#Create a list of lists to hold CNEs in each cluster

		lists = [ [] for c in clusters]
		clusterCount = len(clusters)
	
		#start counting loops
	
		loopCounter += 1

		#for each CNE in the dataset
		for f in CNEs:

		#get the distance between the CNE and the centroid of the first cluster

         
			smallest_distance = getDistance(f,clusters[0].centroid)
		
		#set the cluster this CNE belongs to 
			clusterIndex = 0

			for i in range(clusterCount - 1):

			#calculate distance of the CNE to each other cluster's centroid

				distance = getDistance(f,clusters[i+1].centroid)
				
				#if it's closer to another centroid update 

				if distance < smallest_distance:
					smallest_distance = distance
					clusterIndex = i+1

			lists[clusterIndex].append(f)
		
		biggest_shift = 0.0	
		   # As many times as there are clusters ...
		for i in range(clusterCount):
		    # Calculate how far the centroid moved in this iteration
		    shift = clusters[i].update(lists[i],getDistance)
		    # Keep track of the largest move from all cluster centroid updates
		    biggest_shift = max(biggest_shift, shift)
		
		# If the centroids have stopped moving much, say we're done!
		if biggest_shift == 0.0:	
		    print "Converged after %s iterations" %loopCounter
		    break
	return clusters

		




#**************************************** EUCLIDEAN DISTANCE FUNCTION ***************************************#

def get_E_Distance(a,b):
    
#Euclidean Distance
    if a.n != b.n:
        raise Exception("ILLEGAL: non comparable points")
    
    ret = reduce(lambda x,y: x + pow((a.features[y]-b.features[y]), 2),range(a.n),0.0)
    return math.sqrt(ret)



#**************************************** KL DIVERGENCE FUNCTION *******************************************#

	
def get_K_Distance(f,cluster):
    
	distance = entropy(cluster.features,f.features)
	#print distance
	return distance

	

# INITIALISE MAIN METHOD

if __name__ == "__main__": 
	main()





	





			
	
	








