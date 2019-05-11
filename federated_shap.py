import scipy.special 
import numpy as np
import itertools

#federated_shap methods
class federated_shap():
    def __init__(self):
        pass

    def _powerset(self, iterable):
        s = list(iterable)
        return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

    def _shapley_kernel(self, M ,s):
        if s == 0 or s == M:
            return 10000
        return (M-1)/(scipy.special.binom(M,s)*s*(M-s))



        #Original shap function
    '''
        f: model
        x: one instance with features
        reference: To determine the impact
            of a feature, that feature is set to "missing" and the change in the model output
            is observed. Since most models aren't designed to handle arbitrary missing data at test
            time, we simulate "missing" by replacing the feature with the values it takes in the
            background dataset. So if the background dataset is a simple sample of all zeros, then
            we would approximate a feature being missing by setting it to zero. For small problems
            this background dataset can be the whole training set, but for larger problems consider
            using a single reference value or using the kmeans function to summarize the dataset.
        M: number of features
    '''
    def kernel_shap(self, f, x, reference, M):

        X = np.zeros((2**M,M+1))
        X[:,-1] = 1
        weights = np.zeros(2**M)
        V = np.zeros((2**M,M))
        for i in range(2**M):
            V[i,:] = reference

        ws = {}
        for i,s in enumerate(self._powerset(range(M))):
            s = list(s)
            #print(s)
            V[i,s] = x[s]
            X[i,s] = 1
            ws[len(s)] = ws.get(len(s), 0) + self._shapley_kernel(M,len(s))
            weights[i] = self._shapley_kernel(M,len(s))
        y = f(V)
        tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
        return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))

    #Federated Shap Function
    '''
        f: model
        x: one instance with features
        reference: To determine the impact
            of a feature, that feature is set to "missing" and the change in the model output
            is observed. Since most models aren't designed to handle arbitrary missing data at test
            time, we simulate "missing" by replacing the feature with the values it takes in the
            background dataset. So if the background dataset is a simple sample of all zeros, then
            we would approximate a feature being missing by setting it to zero. For small problems
            this background dataset can be the whole training set, but for larger problems consider
            using a single reference value or using the kmeans function to summarize the dataset.
        M: number of features
        fed_pos: feature position in x start from which the features are hidden and aggregated
    '''
    def kernel_shap_federated(self, f, x, reference, M, fed_pos):
        M_real = M
        M_cur = fed_pos + 1 #with one extra feature as the aggregated hidden features

        X = np.zeros((2**M_cur,M_cur+1))
        X[:,-1] = 1

        weights = np.zeros(2**M_cur)
        V = np.zeros((2**M_cur,M_real))
        for i in range(2**M_cur):
            V[i,:] = reference

        ws = {}

        hidden_index = range(fed_pos, M_real)

        for i,s in enumerate(self._powerset(range(M_cur))):
            #s is the different combinations of features
            s = list(s)
            #print(x)
            #print(s)
            V[i,s] = x[s]
            #if s contains the last combined feature, those hidden features will be set to real values instead of reference
            if fed_pos in s:
                #print(x)
                #print(hidden_index)
                V[i,hidden_index] = x[hidden_index]
            X[i,s] = 1
            ws[len(s)] = ws.get(len(s), 0) + self._shapley_kernel(M_cur,len(s))
            weights[i] = self._shapley_kernel(M_cur,len(s))
        y = f(V)
        tmp = np.linalg.inv(np.dot(np.dot(X.T, np.diag(weights)), X))
        return np.dot(tmp, np.dot(np.dot(X.T, np.diag(weights)), y))



###########Dummy Testing#########################
#Function that imitates the model, takes in instance features and outputs predictons
def f(X):
    np.random.seed(0)
    beta = np.random.rand(X.shape[-1])
    return np.dot(X,beta) + 10

#Original Shap
print("Original Shap Dummy Testing:")
M = 10
np.random.seed(1)
x = np.random.randn(M)
reference = np.zeros(M)
fs = federated_shap()
phi = fs.kernel_shap(f, x, reference, M)
base_value = phi[-1]
shap_values = phi[:-1]

print("  reference =", reference)
print("          x =", x)
print("shap_values =", shap_values)
print(" base_value =", base_value)
print("   sum(phi) =", np.sum(phi))
print("       f(x) =", f(x))

#Federated Shap
print("Federated Shap Dummy Testing:")
M = 10
np.random.seed(1)
x = np.random.randn(M)
reference = np.zeros(M)
fed_pos = 6
fs = federated_shap()
phi = fs.kernel_shap_federated(f, x, reference, M, fed_pos)
base_value = phi[-1]
shap_values = phi[:-1]

print("  reference =", reference)
print("          x =", x)
print("shap_values =", shap_values)
print(" base_value =", base_value)
print("   sum(phi) =", np.sum(phi))
print("       f(x) =", f(x))