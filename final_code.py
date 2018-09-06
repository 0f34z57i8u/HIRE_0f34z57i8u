
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF


# In[3]:


def Lipschitz_W1(X,corrupted_rate,gamma,z):
    term_1 = gamma * X.dot(np.transpose(X))
    term_2 = (1 - corrupted_rate) * (1 - corrupted_rate) * (np.ones([z, z]) - np.diag(np.ones([z]))) * np.dot(X, np.transpose(X))
    term_2 += (1-corrupted_rate) * np.diag(np.ones([z])) * np.dot(X,np.transpose(X))
    return (1/LA.norm(term_1 + term_2, 'fro'))

def Lipschitz_W2(Y,corrupted_rate,gamma,q):
    term_1 = gamma * Y.dot(np.transpose(Y))
    term_2 = (1-corrupted_rate) * (1-corrupted_rate) * (np.ones([q,q]) - np.diag(np.ones([q]))) * np.dot(Y, np.transpose(Y))
    term_2 += (1-corrupted_rate) * np.diag(np.ones([q])) * np.dot(Y, np.transpose(Y))
    return (1/LA.norm(term_1 + term_2, 'fro'))

def Lipschitz_S1(U):
    return (1/LA.norm(np.transpose(U).dot(U), 'fro'))

def Lipschitz_S2(V):
    return (1/LA.norm(np.transpose(V).dot(V), 'fro'))

def Lipschitzz_V1(U1,U2,V2,lamda,beta,Q1,S2,gamma):
    term_1 = LA.norm(np.transpose(V2).dot(np.transpose(U2)).dot(np.transpose(U1)),'fro') * LA.norm(U1.dot(U2).dot(V2),'fro')
    term_2 = lamda
    term_3 = LA.norm(beta*np.transpose(V2).dot(V2),'fro')*LA.norm(Q1.dot(np.transpose(Q1)),'fro')
    term_4 = LA.norm(gamma*np.transpose(V2).dot(np.transpose(S2)).dot(S2).dot(V2),'fro')
    return 1/(term_1+term_2+term_3+term_4)
def Lipschitzz_U1(U2,V2,V1,alpha,lamda,P1,S1,gamma):
    term_1 = LA.norm(U2.dot(V2).dot(V1),'fro') * LA.norm(np.transpose(V1).dot(np.transpose(V2)).dot(np.transpose(U2)),'fro')
    term_2 = LA.norm(alpha*np.transpose(P1).dot(P1),'fro') * LA.norm(U2.dot(np.transpose(U2)),'fro')
    term_3 = lamda
    term_4 = LA.norm(gamma*U2.dot(np.transpose(S1)).dot(S1).dot(np.transpose(U2)),'fro')
    return 1/(term_1+term_2+term_3+term_4)

def Lipschitzz_V2(U1,U2,beta,V1,Q1,lamda,gamma,S2,m1):
    term_1 = LA.norm(np.transpose(U2).dot(np.transpose(U1)),'fro') * LA.norm(U1.dot(U2),'fro') * LA.norm(V1,'fro') * LA.norm(V1.T,'fro')
    term_2 = LA.norm(beta*(np.eye(m1,m1)-V1.dot(Q1)).dot(np.transpose(np.eye(m1,m1)-V1.dot(Q1))),'fro')
    term_3 = lamda
    term_4 = LA.norm(gamma*np.transpose(S2).dot(S2),'fro') * LA.norm(V1.dot(np.transpose(V1)),'fro')
    return 1/(term_1+term_2+term_3+term_4)
def Lipschitzz_U2(U1,V1,V2,lamda,gamma,S1,P1,alpha,n1):
    term_1 = LA.norm(np.transpose(U1),'fro') * LA.norm(U1,'fro') * LA.norm(V2.dot(V1),'fro') * LA.norm(np.transpose(V1).dot(np.transpose(V2)),'fro')
    term_2 = lamda
    term_3 = LA.norm(gamma*np.transpose(U1).dot(U1),'fro') * LA.norm(np.transpose(S1).dot(S1),'fro')
    term_4 = LA.norm(alpha*np.transpose(np.eye(n1,n1)-P1.dot(U1)).dot(np.eye(n1,n1)-P1.dot(U1)),'fro')
    return 1/(term_1+term_2+term_3+term_4)
def SGD_W1(X,corrupted_rate,gamma,S_1,U,z,W1):
    '''
    Params
      Input:
        X: numpy array, dimension(z * n)
          Side information of users
        
        corrupted_rate: float
          corrupte probability of every side information, used to generate \bar{x} and \tlide{x}
          
        gamma: float
        
        S_1: np array, dimension(z * d)
          Projection matrix for U
        
        U: np array, dimension (n * d)
          Latent feature matrix of users
          
        z: dimension of user features
        
      Output:
        matrix, dimension(z * z)
    '''
    U = U.data
    term_1 = (1-corrupted_rate) * np.dot(X,np.transpose(X))
    term_1 += gamma * np.dot(S_1,np.dot(np.transpose(U),np.transpose(X)))
    term_2 = gamma * W1.dot(np.dot(X,np.transpose(X)))
    T_tmp = (1 - corrupted_rate) * (1 - corrupted_rate) * (np.ones([z, z]) - np.diag(np.ones([z]))) * np.dot(X, np.transpose(X))
    T_tmp += (1-corrupted_rate) * np.diag(np.ones([z])) * np.dot(X,np.transpose(X))
    term_2 += W1.dot(T_tmp)
    return term_2 - term_1

def SGD_S1(W_1,X,U,gamma,S_1):
    '''
    Params:
    
      Input:
        W_1: numpy array, dimension(z * z)
          Mapping function for X in auto-coder
        
        X: numpy array, dimension(z * n)
          Side information of users
          
        U: numpy array, dimension(n * d)
          latent features matrix of users
          
      output:
        matrix, dimension(z * d)
    '''
    U = U.data
    a = gamma*S_1.dot(np.transpose(U)).dot(U)
    b = gamma * W_1.dot(X).dot(U)
    return a-b

def SGD_W2(V,corrupted_rate,gamma,S_2,Y,W2,q):
    '''
    Params:
    
      Input:
        Y: numpy array, dimension(q * m)
          Side information of items
        
        corrupted_rate: float
          corrupte probability of every side information, used to generate \bar{x} and \tlide{x}
          
        gamma: float
        
        S_2: np array, dimension(q * d)
          Projection matrix for V
        
        V: np array, dimension (m * d)
          Latent feature matrix of items
          
        q: dimension of item features
    '''
    V = np.transpose(V)
    term_1 = (1-corrupted_rate)*np.dot(Y,np.transpose(Y))
    term_1 += gamma * (S_2.dot(np.transpose(V)).dot(np.transpose(Y)))
    term_2 = gamma * W2.dot(Y).dot(np.transpose(Y))
    T_tmp = (1-corrupted_rate) * (1-corrupted_rate) * (np.ones([q,q]) - np.diag(np.ones([q]))) * np.dot(Y, np.transpose(Y))
    T_tmp += (1-corrupted_rate) * np.diag(np.ones([q])) * np.dot(Y, np.transpose(Y))
    term_2 += W2.dot(T_tmp)
    return term_2-term_1

def SGD_S2(W_2,Y,V,S_2,gamma):
    '''
    Params:
      
      input:
        W_2: numpy array, dimension(q * q)
          Mapping function for Y in auto-coder
        
        Y: numpy array, dimension(q * m)
          Side information of items
          
        V: numpy array, dimension(m * d)
          latent features matrix of items
          
      Output:
        matrix, dimension(q * d)
    '''
    V = V.T
    return gamma*(S_2.dot(np.transpose(V))-W_2.dot(Y)).dot(V)

def SGD_V1(U1,U2,V2,sigma,V1,rating_matrix,lamda,beta,Q1,gamma,S2,Y,W2):
    term_1 = np.transpose(U1.dot(U2).dot(V2)).dot(sigma *(U1.dot(U2).dot(V2).dot(V1)-rating_matrix))
    term_2 = lamda* V1
    term_3 = beta*np.transpose(V2).dot(V2.dot(V1).dot(Q1)-V2).dot(np.transpose(Q1))
    term_5 = gamma*np.transpose(V2).dot(np.transpose(S2)).dot(S2.dot(V2).dot(V1)-W2.dot(Y))
    return term_1+term_2+term_3 + term_5
def SGD_U1(sigma,U1,U2,V1,V2,rating_matrix,alpha,P1,lamda,gamma,S1,X,W1):
    term_1 = (sigma*(U1.dot(U2).dot(V2).dot(V1)-rating_matrix)).dot(np.transpose(U2.dot(V2).dot(V1)))
    term_2 = alpha * np.transpose(P1).dot(P1.dot(U1).dot(U2)-U2).dot(np.transpose(U2))
    term_3 = lamda * U1
    term_4 = gamma*(U1.dot(U2).dot(np.transpose(S1))-np.transpose(X).dot(W1)).dot(S1).dot(np.transpose(U2))
    return term_1+term_2+term_3+term_4
def SGD_V2(U2,U1,sigma,V2,V1,rating_matrix,beta,Q1,m1,lamda,gamma,S2,W2,Y):
    term_1 = np.transpose(U2).dot(np.transpose(U1)).dot(sigma*(U1.dot(U2).dot(V2).dot(V1)-rating_matrix)).dot(np.transpose(V1))
    term_2 = beta*V2.dot(np.eye(m1,m1)-V1.dot(Q1)).dot(np.transpose(np.eye(m1,m1)-V1.dot(Q1)))
    term_3 = lamda*V2
    term_4 = gamma*np.transpose(S2).dot(S2.dot(V2).dot(V1)-W2.dot(Y)).dot(np.transpose(V1))
    return term_1+term_2+term_3+term_4
def SGD_U2(U1,sigma,U2,V2,V1,rating_matrix,beta,gamma,S1,X,W1,alpha,P1,n1):
    term_1 = np.transpose(U1).dot(sigma*(U1.dot(U2).dot(V2).dot(V1)-rating_matrix)).dot(np.transpose(V1)).dot(np.transpose(V2))
    term_2 = lamda*U2
    term_3 = gamma * np.transpose(U1).dot((U1.dot(U2).dot(np.transpose(S1)))-np.transpose(X).dot(W1)).dot(S1)
    term_4 = alpha*(np.transpose(np.eye(n1,n1)-P1.dot(U1))).dot(np.eye(n1,n1)-P1.dot(U1)).dot(U2)
    return term_1+term_2+term_3+term_4
# actual is training_data
# pred is U1U2V2V1
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)
def initialize(z,d,q):
    S1 = np.random.rand(z,d)
    S2 = np.random.rand(q,d)
    W1 = np.random.rand(z,z)
    W2 = np.random.rand(q,q)
    return S1,S2,W1,W2


# In[6]:


# X,Y,P_1,Q_1 has been predefined
X = np.loadtxt('data/ml-100k/X.txt')
Y = np.loadtxt('data/ml-100k/Y.txt')
P_1 = np.loadtxt('data/ml-100k/user_hierarchy.txt')
Q_1 = np.loadtxt('data/ml-100k/item_hierarchy.txt')
z,d,q = X.shape[0], 50,Y.shape[0]
lamda = 1 # regularization index
n1 = P_1.shape[0]
n = P_1.shape[1]
m1 = Q_1.shape[1]
m = Q_1.shape[0]


# In[ ]:


# five fold validation
# first update W1, then W2, then S1, then S2
# when write in function, input gamma, corrupted_rate, S_1, S_2, U, V, z, q, side_info_matrix:user_side_info, movies_side_info
def train_test_result():
	r_cols = ['user_id','movie_id','rating','unix_timestamp']
	for i in range(1,6):
	    train_data = pd.read_csv('data/ml-100k/u{}.base'.format(i),sep = '\t',names=r_cols,
	                            encoding='latin-1')
	    test_data = pd.read_csv('data/ml-100k/u{}.test'.format(i),sep = '\t',names=r_cols,encoding='latin-1')
	    # load train_data
	    print('load the data{}'.format(i))
	    user_record = train_data.user_id.tolist()
	    movie_record = train_data.movie_id.tolist()
	    ratings_record = train_data.rating.tolist()
	    rating_matrix = np.zeros([n,m])
	    sigma_matrix = np.zeros([n,m])
	    sigma = sigma_matrix
	    for i in range(len(user_record)):
	        rating_matrix[user_record[i]-1,movie_record[i]-1] = ratings_record[i]
	        sigma_matrix[user_record[i]-1,movie_record[i]-1] = 1
	    # load test_data
	    user_record_test = test_data.user_id.tolist()
	    movie_record_test = test_data.movie_id.tolist()
	    ratings_record_test = test_data.rating.tolist()
	    rating_matrix_test = np.zeros([n,m])
	    print('data load finish')
	    for i in range(len(user_record_test)):
	        rating_matrix_test[user_record_test[i]-1,movie_record_test[i]-1] = ratings_record_test[i]
	    
	    # record loss for each test_data
	    final_test_loss = []
	    
	    train_loss = {}
	    test_loss = {}
	    good_test_loss = 10000
	    
	    for learning_rate in [1e-6,1e-7,1e-8]:
	        for gamma in [0.3,0.5,0.8]:
	            for corrupted_rate in [0.1,0.2]:
	                for beta in [0.3,0.5,0.8]:
	                    for lamda in [0.3,0.5,0.8]:
	                        for alpha in [0.3,0.5,0.8]:
	                            # split the matrix
	                            S1, S2, W1, W2 = initialize(z,d,q)
	                            
	                            model = NMF(n_components=d, init='random')
	                            U = model.fit_transform(rating_matrix)
	                            V = model.components_

	                            model_2 = NMF(n_components=m1,init = 'random')
	                            V2 = model_2.fit_transform(V)
	                            V1 = model_2.components_

	                            model_3 = NMF(n_components=n1,init='random')
	                            U1 = model_3.fit_transform(U)
	                            U2 = model_3.components_
	                            train_loss_record = []
	                            test_loss_record = []
	                            times = 0
	                            old_loss = 100000
	                            for _ in range(20000):
	                                U = U1.dot(U2)
	                                V = V2.dot(V1)
	                                pred = U.dot(V)
	                                loss = get_mse(pred,rating_matrix)
	                                loss_test = get_mse(pred,rating_matrix_test)
	                                if loss_test > old_loss:
	                                    break
	                                old_loss = loss_test
	                                if times%1000 == 0:
	                                    print('tmp loss {}'.format(loss_test))
	                                
	                                W1 -= Lipschitz_W1(X,corrupted_rate,gamma,z) * SGD_W1(X,corrupted_rate,gamma,S1,U,z,W1)
	                                W2 -= Lipschitz_W2(Y,corrupted_rate,gamma,q) * SGD_W2(V,corrupted_rate,gamma,S2,Y,W2,q)
	                                S1 -= Lipschitz_S1(U) * SGD_S1(W1,X,U,gamma,S1)
	                                S2 -= Lipschitz_S2(V) * SGD_S2(W2,Y,V,S2,gamma)
	                                V1 -= Lipschitzz_V1(U1,U2,V2,lamda,beta,Q_1,S2,gamma) * SGD_V1(U1,U2,V2,sigma,V1,rating_matrix,lamda,beta,Q_1,gamma,S2,Y,W2)
	                                V2 -= Lipschitzz_V2(U1,U2,beta,V1,Q_1,lamda,gamma,S2,m1) * SGD_V2(U2,U1,sigma,V2,V1,rating_matrix,beta,Q_1,m1,lamda,gamma,S2,W2,Y)
	                                U1 -= Lipschitzz_U1(U2,V2,V1,alpha,lamda,P_1,S1,gamma) * SGD_U1(sigma,U1,U2,V1,V2,rating_matrix,alpha,P_1,lamda,gamma,S1,X,W1)
	                                U2 -= Lipschitzz_U2(U1,V1,V2,lamda,gamma,S1,P_1,alpha,n1) * SGD_U2(U1,sigma,U2,V2,V1,rating_matrix,beta,gamma,S1,X,W1,alpha,P_1,n1)
	                                times+=1
	                                train_loss_record.append(loss)
	                                test_loss_record.append(loss_test)
	                        
	                        mse_loss_test = get_mse(pred,rating_matrix_test)
	                        if mse_loss_test < good_test_loss:
	                            good_test_loss = mse_loss_test
	                            train_loss[i] = train_loss_record
	                            print('new good test loss with {}'.format(good_test_loss))
	                            test_loss[i] = test_loss_record
	    final_test_loss.append(good_test_loss)
	    return final_test_loss


if __name__ == "__main__":
	test_loss_final = train_test_result()
	print('final test loss for all batch is:',final_test_loss)