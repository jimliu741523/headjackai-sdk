import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn import utils
from collections import Counter
from headjackai.utils import pandas_check



class svd(object):
    ''' Compute the singular value decomposition of a matrix

    '''    
    def __init__(self, n_features=None):
        '''
        Args:
           n_features(int): number of features for dimension reduction 
           
        Note:
           if n_features=None, it will return full matrix

        '''        
        self.n_features=n_features

    def fit(self, df):
        if self.n_features==None:
            self.n_features = df.shape[1]
            
        self.df = df
        self.sigma = np.zeros((df.shape[0], df.shape[1]))
        self.U, S, self.V = np.linalg.svd(df, full_matrices=True)
        for i in range(self.n_features):
            self.sigma[i][i] = S[i]
            
    def transform(self, df):
        dr_matrix = np.dot(self.U, self.sigma) 
        return dr_matrix[:,:self.n_features]
    
    def recover_matrix(self, dr_matrix, V):
        dr_matrix = np.pad(dr_matrix, ((0,0),(0,self.df.shape[1]-dr_matrix.shape[1])))
        recover_matrix = np.dot(dr_matrix, V)
        return recover_matrix
    
    def v_values(self):
        return self.V
    
    

class k_anonymize_clustering_base(object):
    ''' the class for k-anonymity and l-diversity based on the cluster method

    '''       
    
    def __init__(self, k, l, cat_col):
        '''
        Args:
           k(int): the number of hiding in the crowd guarantee (k-anonymity)
           l(int): the number of the requirement that the values of the sensitive attributes (l-diversity)
           cat_col(list): the list of the category features 

        '''         
    
        self.k = k
        self.l = l
        self.cat_col = cat_col
    
    def fit(self, df):
        pandas_check(df)
        assert df.shape[0]>=self.l, "the number of data less than l"
        assert df.shape[0] >= self.k*self.l, "the l or k are too large"
        
        nbrs = NearestNeighbors(n_neighbors=df.shape[0])
        nbrs.fit(df)
        distances, self.indices = nbrs.kneighbors(df)
        
        self.km = KMeans(self.l, init='k-means++')
        self.km.fit(df)
    
    def transform(self, df):
        pandas_check(df)
        scores = silhouette_samples(df, self.km.labels_)

        df["kMeansLabels"] = self.km.labels_
        df["silhouetteScores"] = scores
        
        ##for t-closeness
        # df = df.sort_values("silhouetteScores")
        df = utils.shuffle(df, random_state=8888)        
        k_list = self.detect_k_ann(df['kMeansLabels'], self.k)
        
        i=0
        while len(k_list)>0:
            re_cluster_ind = df.index[i]
            if df.loc[re_cluster_ind, 'kMeansLabels'] in k_list:
                replace_ind = self.find_replace_cluster_ind(self.indices, re_cluster_ind, k_list, df)
                df.loc[re_cluster_ind, 'kMeansLabels'] = df.loc[replace_ind,'kMeansLabels']
            k_list = self.detect_k_ann(df['kMeansLabels'],self.k)
            i+=1
            
        
        label_df = df['kMeansLabels'].copy()
        df = df.drop(['silhouetteScores','kMeansLabels'],1)
        
        cluster_list = np.unique(label_df)
        for i in range(len(cluster_list)):
            tmp_ind = label_df==cluster_list[i]
            df[tmp_ind] = self.generalization(df[tmp_ind], cat_col=self.cat_col)
            
            
        return df

    @staticmethod
    def find_replace_cluster_ind(indices, re_cluster_ind, k_list, df, j=1):
        replace_cluster_ind = indices[re_cluster_ind][j]
        new_label = df.loc[replace_cluster_ind, 'kMeansLabels']
        old_label = df.loc[re_cluster_ind, 'kMeansLabels']
        if new_label in k_list or new_label ==old_label:
            j+=1
            replace_cluster_ind = k_anonymize_clustering_base.find_replace_cluster_ind(indices, re_cluster_ind, k_list, df,  j)
        return replace_cluster_ind
    
    @staticmethod
    def detect_k_ann(df_cluster, k):
        labels_count = Counter(df_cluster)
        list_labels = list(labels_count.keys())
        k_list = []
        for i in list_labels:
            tmp_count = labels_count[i]
            if 0 < tmp_count < k:
                k_list.append(i)
        return k_list  
    
    @staticmethod
    def generalization(df, cat_col=[]):
        tmp_df = df.copy()
        features_list = tmp_df.columns
        for col in features_list:
            if col not in cat_col:
                tmp_df[col] = np.mean(tmp_df[col])
            else:
                tmp_df[col] = tmp_df[col].value_counts().index[0]
        return tmp_df   
    
    @staticmethod    
    def isKAnonymized(df, k):
        for index, row in df.iterrows():
            query = ' & '.join([f'{col} == {row[col]}' for col in df.columns])
            rows = df.query(query)
            if rows.shape[0] < k:
                return False
        return True
