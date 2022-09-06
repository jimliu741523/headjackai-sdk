import requests
import json
import pandas as pd
from headjackai.utils import pandas_check
from passlib.context import CryptContext


class headjackai_hub(object):
    '''Establish a connection to headjack-ai core server 

    '''
    
    def __init__(self, host):
        '''
        Args:
           host(str): the headjack-ai server URL

        '''
        self.host = host
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
    def knowledge_fit(self, data, target_domain, task_name, label):
        '''Train a knowledge and save on the headjack-ai server 

        Args:
           data(pandas data frame): the data from target domain
           target_domain(str): the name of target knowledge
           task_name(str): the name of the headjack-ai task
           label(str): the name of label columns 

        '''
        
        pandas_check(data)
        info = {'username':self.username,
                   'pwd':self.pwd_context.hash(self.pwd),
                   'target_domain':target_domain,
                   'label':label,
                   'task_name':task_name,
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
                   'pwd':self.pwd_context.hash(self.pwd),
                   'target_domain':target_domain,
                   'source_domain':source_domain,                
                   'label':label,
                   'features_list':json.dumps(features_list),
                   'data':json.dumps(data.to_dict())
                  }
        url = self.host+'/api/knowledge_transform' 
        status = requests.post(url, json=info)
        return pd.DataFrame.from_dict(json.loads(status.json()['jackin_df']))
        

    def fit(self, data, target_domain, task_name, label, val_data=pd.DataFrame(), source_list=['all'], best_domain=True, eval_metric='default', ml_type='lgbm'):
        '''Train a ml pipeline of lightGBM model with headjack features
        
        Args:
           data(pandas data frame): the data from target domain
           target_domain(str): the name of target knowledge
           task_name(str): the name of the headjack-ai task
           label(str): the name of label columns 
           eval_metric: evaluation metrics for validation data, a default metric will be assigned according to objective (mae for regression, and f1 for classification)
           best_domain(boolean): use black-box optimazition or best knowledge in the headjack-ai pipeline
           
        Note:
           1. the black-box optimazition will take a long time to calculate
           2. list of eval_metric: regression of "mae, mse"; classification of "f1, f1_macro, f1_micro,  precision, precision_macro, precision_micro, recall, recall_macro, recall_micro, acc, auc"

        '''
        
        pandas_check(data)      
        pandas_check(val_data)        

        info = {'username':self.username,
                   'pwd':self.pwd_context.hash(self.pwd),
                   'target_domain':target_domain,
                   'task_name':task_name,
                   'best_domain':best_domain,                
                   'label':label,
                   'source_list':json.dumps(source_list),
                   'ml_type':ml_type,
                   'tr_data':json.dumps(data.to_dict()),
                   'ts_data':json.dumps(val_data.to_dict()),
                   'eval_metric': eval_metric
                  }
        url = self.host+'/api/fit' 
        response = requests.post(url, json=info)
        
        return response.json()['status']
        
        

    
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
                   'pwd':self.pwd_context.hash(self.pwd),
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
                   'pwd': self.pwd_context.hash(self.pwd)}
        
        url = self.host+'/api/knowledgepool_check' 
        response = requests.post(url, json=info) 
        return response.json()['knowledge_pool']

    
    def knowledgepool_delete(self, target_domain):
        '''Delete specific knowledge in the headjack-ai server private knowledge pool
        
        Args:
           target_domain(str): the name of target knowledge        
        
        '''         
        
        info = {'username': self.username,
               'pwd': self.pwd_context.hash(self.pwd),
               'target_domain':target_domain}
        
        url = self.host+'/api/knowledgepool_delete' 
        response = requests.post(url, json=info) 
        return response.json()['status']
  

    def account_info_check(self):
        '''Check account information
                  
        '''         
        info = {'username': self.username,
               'pwd': self.pwd_context.hash(self.pwd)}        
        url = self.host+'/api/check_user_info' 
        response = requests.post(url, json=info) 
        return response.json()
    
    
    def fit_status_check(self, task_name, process):
        '''Check fit status
        
        Args:
           task_name(str): the name of the headjack-ai task
           process(str): the process of the status checking
           
        Note:
            the status checking only for func of "fit" and "knowledge_fit"
                  
        ''' 
        
        if process=='knowledge_fit' or  process=='fit':

            info = {'username': self.username,
                   'pwd': self.pwd_context.hash(self.pwd),
                    'task_name': task_name,
                    'process':process}

            url = self.host+'/api/fit_status_check' 
            response = requests.post(url, json=info)              
            
            return response.json()
        else:
            return 'the status checking only for func of "fit" and "fit_knowledge"'
        

    def fit_res_return(self, task_name):
        '''Reutrn fit results
        
        Args:
           task_name(str): the name of the headjack-ai task
           
        ''' 
        

        info = {'username': self.username,
               'pwd': self.pwd_context.hash(self.pwd),
                'task_name': task_name}

        url = self.host+'/api/fit_res_return' 
        response = requests.post(url, json=info) 
        features_list = response.json()['features_list']
        metrics = response.json()['metrics']
        metrics = pd.DataFrame.from_dict(metrics['metrics'])
        

        return features_list, metrics
   
        
        
    def login(self, username, pwd):
        '''login a account to headjack-ai server
        
        Args:
           username(str): the user name for login headjack-ai server
           pwd(str): the passwords for login headjack-ai server
        
        '''        
        
        self.username = username
        self.pwd = pwd  
        info = {'username': username,
                'pwd': self.pwd_context.hash(pwd)}
        url = self.host+'/api/check_user' 
        response = requests.post(url, json=info) 
        
        self.status = response.json()['status']
        if self.status:
            self.username = username
            self.pwd = pwd
            return 'Login Successful!'
        else:
            return 'Please check your username or password'
            

        
