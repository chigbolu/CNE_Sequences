import csv as csv
import sys
import math
import random
import subprocess
import numpy as np
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
from sklearn.datasets import load_digits
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

# We'll generate an animation with matplotlib and moviepy.
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mpy


################################   END IMPORTS    #############################################



	




trainData = csv.reader(open('./flowerData.csv','rb'))
header = trainData.next();

flowerNames = csv.reader(open('./IrisNamesID.csv','rb'))
header2 = flowerNames.next();


# Create an array with the correct cluster results
flowerIDs = [value for value in flowerNames]
flowerIDs = np.array(flowerIDs)

# Convert flowerIDs to list of int
flowerIDs = flowerIDs.astype(int)
flowerIDsL = []	
for f in flowerIDs:
	flowerIDsL.append(f[0])


# Create an array containing the training data
data = []
for row in trainData:
	data.append(row)
data = np.array(data);
index = 0;

# Cast string data to float
floatData = np.array(data, dtype=float)

normData = []

# Normalise data to probability distribution values. Each rows sums 1
for row in floatData:
	thisRow = []
	index = 0
	sumRow = sum(row)
	while index < 4:	
		thisRow.append(row[index]/sumRow)
		index +=1
	normData.append(thisRow)











#************************************************* MAIN METHOD *****************************************************#

#prevent data loss when exception is raised --> store in global variabl outside main

overallPerformance = {'E':0,'KL':0,'bestkKL':0, 'bestkE':0}
thisK = 3
iterate = 0
targetK = []
targetE = []


def main():


	flowers = []
	for row in normData:
		flowers.append(Flower(row))

	global thisK 
	global overallPerformance 
	#store best cluster results
	global targetK
	global targerE
	#global contains the iteration instance , even when a main() exception is raised
	global iterate
	#store best clusters results so far
	thisTargetE = []
	thisTargetK = []
	print "Running k-Means  -  k = ",thisK 
	while(iterate < 20):
		
	
		# Calculate clusters applying Kmeans with Euclidean distance and Kmeans with KL distance
			clustersE = kmeans(flowers, thisK, get_E_Distance)
			clustersK = kmeans(flowers, thisK, get_K_Distance)


		# Obtain cluster values (0,1,2) in order to measure clustering similarity
	
			clusterValuesE = []
			clusterValuesK = []
			i = 0


			while i < thisK:
		
				flowersE = clustersE[i].flowers
				flowersK = clustersK[i].flowers
				for f in flowersE:
					clusterValuesE.append(i)
				for f in flowersK:
					clusterValuesK.append(i)
				i += 1




		# Print out the results of the sklearn Rand Index function
				
			#rand_measures = adjusted_rand_score(clusterValuesK,clusterValuesE)
			rand_KL = adjusted_rand_score(clusterValuesK,flowerIDsL)
			rand_E = adjusted_rand_score(clusterValuesE,flowerIDsL)
			score_KL = silhouette_score(normData, clusterValuesK, sample_size=len(normData))
			score_E =  silhouette_score(normData, clusterValuesE, sample_size=len(normData))
				

	
			if(score_KL > overallPerformance['KL'] ):
				overallPerformance['KL'] = rand_KL
				overallPerformance['bestkKL'] = thisK
				#overallPerformance['Rand_ScoreKL'] = rand_KL
				targetK = clusterValuesK
				#print "This Rand = ", overallPerformance['KL']
				#print "This Rand = ", overallPerformance['Rand_ScoreKL']
			if(score_E > overallPerformance['E']):
				overallPerformance['E'] = rand_E
				overallPerformance['bestkE'] = thisK
				#overallPerformance['Rand_ScoreE'] = rand_E
				targetE = clusterValuesE
				#print "This Rand = ", overallPerformance['E']
				#print "This Rand = ", overallPerformance['Rand_ScoreE']
		
			iterate += 1

	
	
	
	print "KL: The best k = ", overallPerformance['bestkKL'], "The rand score = ", overallPerformance['KL']
	print "E: The best k = ", overallPerformance['bestkE'], "The best rand score = ", overallPerformance['E']



	############           PLOT GRAPH        ################		


	### Plot KL graph#####

	#map flowers names
	names_map = {0:"I-s",1:"I-ver",2:"I-vir"}
	normData2 = np.array(normData)
	normData2.shape
	X = np.vstack([normData2])
	y = np.hstack([targetK])

	tsne_Calc = TSNE(random_state=RS).fit_transform(X)

	def scatter(x, colors):
	    # We choose a color palette with seaborn.
	    palette = np.array(sns.color_palette("hls", 10))

	    # We create a scatter plot.
	    f = plt.figure(figsize=(12, 12))
	    ax = plt.subplot(aspect='equal')
	    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
		            c=palette[colors.astype(np.int)])
	    plt.xlim(-25, 25)
	    plt.ylim(-25, 25)
	    ax.axis('off')
	    ax.axis('tight')


	  #We add the labels for each digit.
	    txts = []
	    for i in range(3):
            # Position of each label.
	   	xtext, ytext = np.median(x[colors == i, :], axis=0)
            	txt = ax.text(xtext, ytext, names_map[i], fontsize=18)
            	txt.set_path_effects([
            		PathEffects.Stroke(linewidth=5, foreground="w"),
            		PathEffects.Normal()])
            	txts.append(txt)

    	    return f, ax, sc, txts



	
	scatter(tsne_Calc, y)
	plt.savefig('images/irisKL_graph.png', dpi=200)




	##### Plot Euclidean graph #####

	y = np.hstack([targetE])

	tsne_Calc = TSNE(random_state=RS).fit_transform(X)

	def scatter(x, colors):
	    # We choose a color palette with seaborn.
	    palette = np.array(sns.color_palette("hls", 10))

	    # We create a scatter plot.
	    f = plt.figure(figsize=(12, 12))
	    ax = plt.subplot(aspect='equal')
	    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
		            c=palette[colors.astype(np.int)])
	    plt.xlim(-25, 25)
	    plt.ylim(-25, 25)
	    ax.axis('off')
	    ax.axis('tight')


	#We add the labels for each digit.
	    txts = []
	    for i in range(3):
            # Position of each label.
	   	xtext, ytext = np.median(x[colors == i, :], axis=0)
            	txt = ax.text(xtext, ytext, names_map[i], fontsize=18)
            	txt.set_path_effects([
            		PathEffects.Stroke(linewidth=5, foreground="w"),
            		PathEffects.Normal()])
            	txts.append(txt)

    	    return f, ax, sc, txts


	scatter(tsne_Calc, y)
	plt.savefig('images/irisE_graph.png', dpi=200)



	








	


#******************************** FLOWER CLASS ****************************************#


class Flower:
	#a flower is described by 4 features
	def __init__(self,features):
	
		self.features = features
		self.n = len(features)



	def __repr__(self):
 	       return str(self.features)







#******************************* CLUSTER CLASS ***************************************#

class Cluster:
	def __init__(self,flowers):
		if len(flowers) == 0: raise Exception("ILLEGAL: empty cluster")
	
		#the features that belong to this cluster
		self.flowers = flowers
		self.n = flowers[0].n
		self.centroid = self.calculateCentroid()

		 # The number of featues
		

	def __repr__(self):
       
     #   String representation of this object
       
		return str(self.flowers)

	def update(self,flowers,getDistance):

		#returns the distance between previous and the new centroid after recalculating and storing the new centroid

		old_centroid = self.centroid
		self.flowers = flowers
		self.centroid = self.calculateCentroid()
		
	        shift = getDistance(old_centroid,self.centroid)
		return shift

	def calculateCentroid(self):
	
		numflowers = len(self.flowers)
		features = [f.features for f in self.flowers]
		# Reformat that so all x's are together, all y'z etc.
		unzipped = zip(*features)
		# Calculate the mean for each dimension

		centroid_features = [math.fsum(dList)/numflowers for dList in unzipped]

		return Flower(centroid_features)


#***************************** KMEANS FUNCTION ****************************************#


def kmeans(flowers, k, getDistance):

         
	initial = random.sample(flowers, k)
	clusters = [Cluster([f]) for f in initial]

	loopCounter = 0
	
	while True:
	
		#Create a list of lists to hold flowers in each cluster

		lists = [ [] for c in clusters]
		clusterCount = len(clusters)
	
		#start counting loops
	
		loopCounter += 1

		#for each flower in the dataset
		for f in flowers:

		#get the distance between the flower and the centroid of the first cluster

         
			smallest_distance = getDistance(f,clusters[0].centroid)
		
		#set the cluster this flower belongs to 
			clusterIndex = 0

			for i in range(clusterCount - 1):

			#calculate distance of the flower to each other cluster's centroid

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
	i = 0
	finish = False
	while not finish:
		try:
			main()
			finish = True
			
		except:
			print "Exception raised n. ", i
			finish = False

		i += 1
#	main()	
	



















