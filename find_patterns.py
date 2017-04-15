
######### DATA IMPORTS #############

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
from sklearn.metrics import *

######## PLOT TSNE GRAPH IMPORTS
# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
#%matplotlib inline

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
import imageio

from numpy.linalg import norm



################################   END IMPORTS    #############################################


additive_smoothing = 0.0000001
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
	#apply additive smoothing on data in order to avoid divisions by 0
	cne_list.insert(i, s, additive_smoothing)
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


#************************************************* MAIN METHOD *****************************************************#

overallPerformance = {'E':-5,'KL':-5,'bestkKL':0, 'bestkE':0}
thisK = 2
iterate = 0
targetK = []
targetE = []
coord_label = {}


def main():

	global thisK 
	global overallPerformance 

	while thisK < 3: 

		global iterate 
		global targetK
		global targerE
		print "Running k-Means  -  k = ",thisK 

		while iterate < 3:
			cnes_objs = []
			for row in cne_data:
				cnes_objs.append(CNE(row))

		# Calculate clusters applying Kmeans with Euclidean distance and Kmeans with KL distance

			print "Running k-means / E .................  -- k =",thisK

			clustersE = kmeans(cnes_objs, thisK, get_E_Distance)

			print "Running k-means / KL.................  -- k =",thisK
			clustersK = kmeans(cnes_objs, thisK, get_K_Distance)


		# Obtain cluster values k in order to measure clustering similarity
	

			clusterValuesE = []
			clusterValuesK = []

			i = 0
			countInstancesE = [0] * thisK
			countInstancesK = [0] * thisK
			while i < thisK:
		
				CNEsE = clustersE[i].CNEs
				CNEsK = clustersK[i].CNEs
				

				for f in CNEsE:
					clusterValuesE.append(i)
					countInstancesE[i] += 1
				for f in CNEsK:
					clusterValuesK.append(i)
					countInstancesK[i] += 1
				i += 1
	



			#silhouette calculation
			score_KL =  silhouette_score(cne_data, clusterValuesK, sample_size= len(cne_data))
			score_E =   silhouette_score(cne_data, clusterValuesE, sample_size=len(cne_data))



			if(score_KL > overallPerformance['KL'] ):
				overallPerformance['KL'] = score_KL
				overallPerformance['bestkKL'] = thisK
				targetK = clusterValuesK
				
				print "This Sill = ", overallPerformance['KL']
					
			if(score_E > overallPerformance['E']):
				overallPerformance['E'] = score_E
				overallPerformance['bestkE'] = thisK
				targetE = clusterValuesE
				
				print "This Sill = ", overallPerformance['E']
			
			iterate += 1


			

		thisK += 1
		iterate = 0
		

	print "KL: The best k = ", overallPerformance['bestkKL'], "The best silhouette = ", overallPerformance['KL']
	print "E: The best k = ", overallPerformance['bestkE'], "The best silhouette = ", overallPerformance['E']




	### Plot KL graph#####

	#for row in targetK:
	#	print row 
	
	cne_data2 = np.array(cne_data)
	cne_data2.shape
	X = np.vstack([cne_data2])
	y = np.hstack([targetK])

	tsne_JSD = TSNE(random_state=RS,metric=JSD).fit_transform(X)
	tsne_E = TSNE(random_state=RS).fit_transform(X)
	tsne_KL = TSNE(random_state=RS,metric=JSD).fit_transform(X)



	######graph points############

	
	# get only first few digits of coordinates	
	def format_coord(a,b):
		a = str(a)[:8]
	        b = str(b)[:8]
		return a + "," + b
		

	def onpick3(event):
	  global coord_label
   	  index = event.ind
    	  xy = event.artist.get_offsets()
   	  print '--------------'
	  a = xy[index].tolist()
	  a = str(a)
	  # format coordinates 
	  a_split = a.split(",")
	  a_1 = a_split[0].replace("[[","")
	  a_2 = a_split[1].replace("]]","")
	  a_2 = a_2.replace(" ","")
	  coord = format_coord(a_1,a_2)
	  #print coord_label[coord]
	  print seq_dict[coord_label[coord]]
	  
	
	
	
	
	def scatter(x, colors):
	    global coord_label
	    labels_in_graph = {}
	    cne_binds = defaultdict(list)
	    # We choose a color palette with seaborn.
	    palette = np.array(sns.color_palette("hls", 10))
	    # We create a scatter plot.
	    f = plt.figure(figsize=(12, 12))
	    ax = plt.subplot(aspect='equal')
	    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
		            c=palette[colors.astype(np.int)],picker=True)
	    plt.xlim(-25, 25)
	    plt.ylim(-25, 25)
	    ax.axis('off')
	    ax.axis('tight')
	    f.canvas.mpl_connect('pick_event', onpick3)

	  #add the labels for each cne
	    txts = []
	    for i in range(overallPerformance['bestkKL']):
            # Position of each label.
	   	xtext, ytext = np.median(x[colors == i, :], axis=0)
            	txt = ax.text(xtext, ytext, str(i), fontsize=18)
            	txt.set_path_effects([
            		PathEffects.Stroke(linewidth=5, foreground="w"),
            		PathEffects.Normal()])
            	txts.append(txt)
	    
	    #store coordinates and cne in dictionary
	    for label,x,y in zip(cne_names,x[:,0],x[:,1]):
		coor_xy = format_coord(x,y)
	        coord_label[coor_xy] = label
		symbols_bound = seq_dict[coord_label[coor_xy]]
	        if(str(symbols_bound) not in labels_in_graph):
	    		plt.annotate(symbols_bound,xy = (x,y), xytext = None,textcoords = None, bbox = None, arrowprops = None, size=5)
			labels_in_graph[str(symbols_bound)] = 1
		else:
			labels_in_graph[str(symbols_bound)] += 1
		cne_binds[str(symbols_bound)].append(label)

          #  print labels_in_graph
	  #  print "-------------------------------------------------------"
	  #  print cne_binds
	    with open('groups_cne.csv', 'wb') as csv_file:
    		writer = csv.writer(csv_file)
    		for key, value in labels_in_graph.items():
       			writer.writerow([key, value])
	        
		
	    with open('cne_binds.csv', 'wb') as csv_file:
    		writer = csv.writer(csv_file)
    		for key, value in cne_binds.items():
       			writer.writerow([key, value])
	        
		

    	    return f, ax, sc, txts

	
	#scatter(tsne_E, y)
	#scatter(tsne_KL,y)
	scatter(tsne_JSD,y)
	plt.savefig('images/cne_graphJSD.png', dpi=600)
	#plt.show()
	








	


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
		if(self.n == 0 ):
			print "**************No features belong to this cluster**********************	"
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
		  #  print "Converged after %s iterations" %loopCounter
		    break
	return clusters

		




#**************************************** EUCLIDEAN DISTANCE FUNCTION ***************************************#

def get_E_Distance(a,b):
    
#Euclidean Distance

    if(a.n == 0 or b.n ==0):
        print "******************No features belong to this cluster in get_E_Distance***************************"
	

    if a.n != b.n:
	#main()
        raise Exception("ILLEGAL: non comparable points")
	
    
    ret = reduce(lambda x,y: x + pow((a.features[y]-b.features[y]), 2),range(a.n),0.0)
    return math.sqrt(ret)



#**************************************** KL DIVERGENCE FUNCTION *******************************************#

	
def get_K_Distance(cne,cluster):
        if(cne.n == 0 or cluster.n == 0):
           print "***********No features belong to this cluster in get_K_Distance*******************"    
	   

	distance = entropy(cluster.features,cne.features)
	return distance

	

def KL_Distance_Graph(a,b):

	distance = entropy(a,b)
	return distance

def JSD(a,b):
	a = a / norm(a, ord=1)
	b = b / norm(b, ord=1)
    	m = 0.5 * (a + b)
    	return 0.5 * (entropy(a, m) + entropy(b, m))



# INITIALISE MAIN METHOD

if __name__ == "__main__": 
#	i = 0
#	finish = False
#	while not finish:
#		try:
#			main()
#			finish = True
#			
##		except:
#			print "Exception raised n. ", i
#			finish = False
#
#		i += 1

	main()



	




	





			
	
	








