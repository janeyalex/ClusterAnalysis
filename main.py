from dataclasses import dataclass
from pickle import TRUE
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean
import numpy as np
import copy
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import plotly.express as px
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from yellowbrick.cluster import SilhouetteVisualizer
from scipy.stats import f_oneway, tukey_hsd

# Code used to download data from github. Once downloaded, just read in the file 
# url = 'https://github.com/dscig/sleeps_clustering/raw/master/data/Data.csv'
# res = requests.get(url, allow_redirects=True)
# with open('sleepData.csv','wb') as file:
#     file.write(res.content)

df = pd.read_csv('sleepData.csv')
copydf = copy.deepcopy(df)
normDf = pd.read_csv('normalizedDataNEW.csv')

# Dataframe without date variables 
nodateDf = copydf[['userId','sleep_start_time', 'sleep_end_time' , 'sleep_min', 'sleep_efficiency', 'awaken_min', 'awaken_moments',
       'nap_count', 'total_nap_min', 'cal_consume', 'active_cal', 'walks',
       'distance', 'stairs', 'active_ratio']]

def eda():
       print(df.columns)

       users = nodateDf.groupby('userId').mean()

       params = {'axes.titlesize':'8',}
       plt.rcParams.update(params)

       # Histograms 
       users.hist(column=['sleep_start_time', 'sleep_end_time',
       'sleep_min', 'sleep_efficiency', 'awaken_min', 'awaken_moments',
       'nap_count', 'total_nap_min', 'cal_consume', 'active_cal', 'walks',
       'distance', 'stairs', 'active_ratio'], xlabelsize= 5, ylabelsize=5)
       plt.show()

       # Correlation plot
       corData = users[['sleep_start_time', 'sleep_end_time',
       'sleep_min', 'sleep_efficiency', 'awaken_min', 'awaken_moments',
       'nap_count', 'total_nap_min', 'cal_consume', 'active_cal', 'walks',
       'distance', 'stairs', 'active_ratio']]

       corr_matrix = corData.corr()
       sns.set(font_scale=0.65)
       plt.figure(figsize=(8,6))
       heatmap = sns.heatmap(corr_matrix, annot=True)
       heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
       plt.show()

       # Pairplot 
       g = sns.pairplot(corData)
       g.fig.set_size_inches(15,15)
       plt.show()



def KmeansClustering(data, k ):
       # K means clustering that takes in data and values of k 
       # Returns cluster labels and prints statistics about clsuters

       kmeans = KMeans(init="k-means++",n_clusters=k,n_init='auto',max_iter=300,random_state=42)
       kmeans.fit(data)
       clusters = kmeans.predict(data)
       print("SSE: ", kmeans.inertia_)
       print("Silhouette Scores: ", silhouette_score(data, kmeans.labels_))

       ch_score = calinski_harabasz_score(data, kmeans.labels_)
       print("Calinski-Harabasz index:", ch_score)

       return clusters
       
def numClusters(data):
       # This function takes in data and plots elbow plots for k- means clustering
       sse = []
       for k in range(1, 11):
              kmeans =  KMeans(init="k-means++",n_clusters=k,n_init='auto',max_iter=300,random_state=42)
              kmeans.fit(data)
              sse.append(kmeans.inertia_)

       plt.plot(range(1, 11), sse)
       plt.xticks(range(1, 11))
       plt.xlabel("Number of Clusters")
       plt.ylabel("SSE")
       plt.title("SSE by Number of Clusters")
       plt.show()

       model = KMeans(init="k-means++",n_init='auto',max_iter=300,random_state=42)
       visualizer = KElbowVisualizer(model, k=(1,12)).fit(data)
       visualizer.show()

def mySilkScore(data):
       # Silhoutte Score visualization 
       for i in [2, 3, 4, 5]:

              km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
              q, mod = divmod(i, 2)
              
              visualizer = SilhouetteVisualizer(km, colors='yellowbrick')
              visualizer.fit(data) 

              visualizer.show()       

def numOfComps(data):
       # Determines the number of components for PCA and plots a visualization
       pca = PCA()
       pca.fit(data)

       plt.figure(figsize=(10,8))
       plt.plot(range(1,16), pca.explained_variance_ratio_.cumsum(), marker='o')
       plt.xlabel('Number of Components', fontsize = 20)
       plt.ylabel("Cumulative Explained Variance", fontsize = 18)
       plt.title("Cumulative Explained Variance with Principal Components", fontsize = 20)
       plt.show()

def myPCA(data, numComps):
       # Takes in data and the number of components and performs PCA
       # returns the scores 
       pca = PCA(n_components=numComps)
       pca.fit(data)
       scores_pca = pca.transform(data)
       return scores_pca

def pcaViz(components, data):
       # Fucntion takes in the first two components of the data and plots a scatterplot colored by cluster
       fig = px.scatter(components, x=0, y=1, color=data['clusters'], 
       title = "First Two Principal Components" ).update_layout(
       xaxis_title="PC 1", yaxis_title="PC2")
       fig.show()

def highClustering(data):
       # Performs hierarchical clsutering with differnt linkage methods
       # Prints statistcs and returns cluster labels
       complete_clustering = linkage(data, method="complete", metric="euclidean")
       average_clustering = linkage(data, method="average", metric="euclidean")
       single_clustering = linkage(data, method="single", metric="euclidean")

       # dendrograms
       dendrogram(complete_clustering)
       plt.title("Dendrogram for Summary Dataset")
       plt.show()

       dendrogram(average_clustering)
       plt.show()

       dendrogram(single_clustering)
       plt.show()

       clusters = cut_tree(complete_clustering, n_clusters=4).reshape(-1, )


       print("Silhouette Scores: ", silhouette_score(data, clusters))

       ch_score = calinski_harabasz_score(data, clusters)
       print("Calinski-Harabasz index:", ch_score)

       return clusters

def postAnalysis():
       # This function plots boxplots for every variable for the three different groups
       fig, axes = plt.subplots(2,3) # create figure and axes
       for i,el in enumerate(list(meanDfnotNorm.columns.values)[3:9]):
              a = meanDfnotNorm.boxplot(el, by="clusters", ax=axes.flatten()[i])
       plt.tight_layout() 
       plt.show()

       fig, axes = plt.subplots(2,3) # create figure and axes

       for i,el in enumerate(list(meanDfnotNorm.columns.values)[9:15]):
              a = meanDfnotNorm.boxplot(el, by="clusters", ax=axes.flatten()[i])
       plt.tight_layout() 
       plt.show()

       fig, axes = plt.subplots(1,2, sharey= True) # create figure and axes
       for i,el in enumerate(list(meanDfnotNorm.columns.values)[1:3]):
              a = meanDfnotNorm.boxplot(el, by="clusters", ax=axes.flatten()[i])
       plt.tight_layout() 
       plt.show()

       # Create seperate dataframes for the three groups
       all_groups = meanDfnotNorm.groupby('clusters')    
       group0 = [all_groups.get_group(x) for x in all_groups.groups][0]
       group1 = [all_groups.get_group(x) for x in all_groups.groups][1]
       group2 = [all_groups.get_group(x) for x in all_groups.groups][2]

       # print(meanDfnotNorm.columns)
       variables = meanDfnotNorm.columns[1:15]

       # Perform One-way ANOVA and Tukey test for each variable 
       for i in variables:
              print(i,':')
              print(f_oneway(group0[i],group1[i], group2[i]))
              print(tukey_hsd(group0[i], group1[i], group2[i]))
              print('')


def doAllClustering(standData, originalDf, kmeans):
       # Main function the will do PCA, clustering method and post analysis 

       ################## PCA ##################

       # determine # of components that explains the most variance
       numOfComps(standData)

       # perfom pca with num of comps decided from above (4 comps for mean DF, 7 for all stats Df)
       scores_pca = myPCA(standData, 7)

       # get df with just first 2 principal comps for visualization 
       twoComps = scores_pca[:,:2]

       
       if(kmeans):
       ############ K Means CLustering #########
              # determine # of clusters to use with pca scores
              numClusters(scores_pca)
              mySilkScore(scores_pca)
              # cluster with k means and get cluster nuber for each individual
              clusters = KmeansClustering(scores_pca, 3)

       else:
       ############ Hierarchical CLustering #########
              clusters = highClustering(scores_pca)

       # add cluster values to orginal mean df 
       originalDf['clusters'] = clusters 
       originalDf["clusters"] = originalDf["clusters"].astype(str)

       # visualize first two PCs
       pcaViz(twoComps,originalDf)

       ############ Post Analysis #################
       postAnalysis()




############## NORMALIZATION #############

# remove -1's and replace with NaN
nodateDf.replace(-1, np.nan, inplace=True)
# take original dataset and group by user and get mean for each variable (ignore NaNs)
meanDfnotNorm = nodateDf.groupby('userId', as_index=False).mean()

# Group by user and get standard deviations
sdDfnotNorm = nodateDf.groupby('userId', as_index=False).std()
sdDfnotNorm.columns = ['userId', 'sd_sleep_start_time', 'sd_sleep_end_time', 'sd_sleep_min',
       'sd_sleep_efficiency', 'sd_awaken_min', 'sd_awaken_moments', 'sd_nap_count',
       'sd_total_nap_min', 'sd_cal_consume', 'sd_active_cal', 'sd_walks', 'sd_distance',
       'sd_stairs', 'sd_active_ratio']

# Group by user and get lower quantile 
lowQnotNorm = nodateDf.groupby('userId', as_index=False).quantile(0.25)
lowQnotNorm.columns = ['userId', 'lowQ_sleep_start_time', 'lowQ_sleep_end_time', 'lowQ_sleep_min',
       'lowQ_sleep_efficiency', 'lowQ_awaken_min', 'lowQ_awaken_moments', 'lowQ_nap_count',
       'lowQ_total_nap_min', 'lowQ_cal_consume', 'lowQ_active_cal', 'lowQ_walks', 'lowQ_distance',
       'lowQ_stairs', 'lowQ_active_ratio']

# Group by user and get upper quantile 
upQnotNorm = nodateDf.groupby('userId', as_index=False).quantile(0.75)
upQnotNorm.columns = ['userId', 'upQ_sleep_start_time', 'upQ_sleep_end_time', 'upQ_sleep_min',
       'upQ_sleep_efficiency', 'upQ_awaken_min', 'upQ_awaken_moments', 'upQ_nap_count',
       'upQ_total_nap_min', 'upQ_cal_consume', 'upQ_active_cal', 'upQ_walks', 'upQ_distance',
       'upQ_stairs', 'upQ_active_ratio']

# Merge all into one dataframe
meanSdnotNorm = meanDfnotNorm.merge(sdDfnotNorm)
meanSdLow = meanSdnotNorm.merge(lowQnotNorm)
allStatsDf = meanSdLow.merge(upQnotNorm)



# perform standard scaler on mean df 
scaler = StandardScaler()
standData = scaler.fit_transform(meanDfnotNorm)   

# perform standard scaler on all stats df 
standAllStats = scaler.fit_transform(allStatsDf) 






# Main Calls

eda()
# K means clustering for mean and summary dataset
doAllClustering(standData,meanDfnotNorm, TRUE)
doAllClustering(standAllStats,allStatsDf, TRUE)

# Hierarchical clustering for mean and summary dataset
doAllClustering(standData,meanDfnotNorm, False)
doAllClustering(standAllStats,allStatsDf, False)

