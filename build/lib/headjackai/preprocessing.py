import numpy as np

class svd(object):
    ''' Compute the singular value decomposition of a matrix

    '''    
    def __init__(self, n_features=None):
        self.n_features=n_features
        '''
        Args:
           n_features(int): number of features for dimension reduction 
           
        Note:
           if n_features=None, it will return full matrix

        '''        
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
        recover_matrix = np.dot(dr_matrix, self.V)
        return recover_matrix
    
    def v_values(self):
        return self.V