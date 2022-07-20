import requests
import json
import pandas as pd
from headjackai.utils import pandas_check
 

class headjackai_hub(object):
    '''Establish a connection to headjack-ai core server 

    '''
    
    def __init__(self, host):
        '''
        Args:
           host(str): the headjack-ai server URL

        '''
        self.host = host

    def knowledge_fit(self, data, target_domain, label):
        '''Train a knowledge and save on the headjack-ai server 

        Args:
           data(pandas data frame): the data from target domain
           target_domain(str): the name of target knowledge
           label(str): the name of label columns 

        '''
        
        pandas_check(data)
        info = {'username':self.username,
                   'pwd':self.pwd,
                   'target_domain':target_domain,
                   'label':label,
                   'data':json.dumps(data.to_dict())
                  }
        url = self.host+'/api/knowledge_fit' 
        status = requests.post(url, json=info)
        return status.json()['status']
                 
    def knowledge_transform(self, data, target_domain, label="", source_domain=None, features_list=None):
        '''Jack-in the headjack features from pretrained knowledge into target domain

        Args:
           data(pandas data frame): the data from target domain
           target_domain(str): the name of target knowledge
           label(str): the name of label columns 
           source_domain(str): the name of source knowledge
           features_list(list): the features list of the headjack-ai 
           
        Note:
           Choose one of source_domain and features_list to use

        '''
        
        pandas_check(data)        
        info = {'username':self.username,
                   'pwd':self.pwd,
                   'target_domain':target_domain,
                   'source_domain':source_domain,                
                   'label':label,
                   'features_list':json.dumps(features_list),
                   'data':json.dumps(data.to_dict())
                  }
        url = self.host+'/api/knowledge_transform' 
        status = requests.post(url, json=info)
        return pd.DataFrame.from_dict(json.loads(status.json()['jackin_df']))
        

    def fit(self, data, target_domain, task_name, label, best_domain=True):
        '''Train a ml pipeline of lightGBM model with headjack features
        
        Args:
           data(pandas data frame): the data from target domain
           target_domain(str): the name of target knowledge
           task_name(str): the name of the headjack-ai task
           label(str): the name of label columns 
           best_domain(boolean): use black-box optimazition or best knowledge in the headjack-ai pipeline
           
        Note:
           the black-box optimazition will take a long time to calculate

        '''
        
        pandas_check(data)        
        info = {'username':self.username,
                   'pwd':self.pwd,
                   'target_domain':target_domain,
                   'task_name':task_name,
                   'best_domain':best_domain,                
                   'label':label,
                   'data':json.dumps(data.to_dict())
                  }
        url = self.host+'/api/fit' 
        response = requests.post(url, json=info)
        
        return response.json()['features_list'], pd.DataFrame.from_dict(json.loads(response.json()['metrics']))

    
    def transform(self, data, target_domain, task_name, label, features_list, proba_domain=False):
        '''Get the prediction reslut from pretrained headjack-ai pipeline
        
        Args:
           data(pandas data frame): the data from target domain
           target_domain(str): the name of target knowledge
           task_name(str): the name of the headjack-ai task
           label(str): the name of label columns 
           features_list(list): the features list of the headjack-ai 
           proba_domain(boolean): the prediction table displays the probability or not  
           
        Note:
           the proba_domain please set False if the task is regression.          
        
        '''   
        
        pandas_check(data)
        info = {'username':self.username,
                   'pwd':self.pwd,
                   'target_domain':target_domain,
                   'task_name':task_name,
                   'label':label,
                   'data':json.dumps(data.to_dict()),
                   'features_list':json.dumps(features_list),
                   'proba_domain':proba_domain
                  }
        url = self.host+'/api/transform' 
        response = requests.post(url, json=info)
        
        return pd.DataFrame.from_dict(json.loads(response.json()['preds_df']))

    
    def knowledgepool_check(self, public_pool=False):
        '''Check list of knowledge pool in the headjack-ai server
        
        Args:
           public_pool(boolean): check the knowledge pool in the headjack-ai or not        
        
        '''    

        if public_pool:
            info = {'username': 'admin'}
        else:
            info = {'username': self.username,
                   'pwd': self.pwd}
        
        url = self.host+'/api/knowledgepool_check' 
        response = requests.post(url, json=info) 
        return response.json()['knowledge_pool']

    
    def knowledgepool_delete(self, target_domain):
        '''Delete specific knowledge in the headjack-ai server private knowledge pool
        
        Args:
           target_domain(str): the name of target knowledge        
        
        '''         
        
        info = {'username': self.username,
               'pwd': self.pwd,
               'target_domain':target_domain}
        
        url = self.host+'/api/knowledgepool_delete' 
        response = requests.post(url, json=info) 
        return response.json()['status']
  

    def login(self, username, pwd):
        '''login a account to headjack-ai server
        
        Args:
           username(str): the user name for login headjack-ai server
           pwd(str): the passwords for login headjack-ai server
        
        '''        
        
        self.username = username
        self.pwd = pwd  
        
        info = {'username': username,
                'pwd': pwd}
        url = self.host+'/api/check_user' 
        response = requests.post(url, json=info) 
        
        self.status = response.json()['status']
        if self.status:
            self.username = username
            self.pwd = pwd
            return 'Login Successful!'
        else:
            return 'Please check your username or password'
            

        
