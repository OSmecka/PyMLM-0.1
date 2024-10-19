import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SIZE_PARAM = #Size used for learning dataset
DISPERSION_PARAM = #Dispersion of the data around fixed linear curve used for learning
X_SIZE = #Size of the x-axis
X_DIV = #Number of decimal places used for learning
X_DIV = X_SIZE/(10**((-1)*X_DIV)
NUM_TRAIN_SETS = #Number of learning itterartions
MAX_NUM = X_SIZE

INPUT_DF = #Table of values for fitting

#-------Define model functions
class CustomLinearRegression:
    def __init__(self):
        self.coef_ = None 
        self.intercept_ = None 

    def fit(self, X_list, y_list):
        X_b_list = [np.c_[np.ones((X.shape[0], 1)), X] for X in X_list]
        self.theta_list = []

        for X_b, y in zip(X_b_list, y_list):
            theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.theta_list.append(theta)
            
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)   
        self.intercept_ = [theta[0] for theta in self.theta_list]
        self.coef_ = [float(coeff) for coeff in theta[1:]]
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        return X_b.dot(self.theta)

    def score(self, y_test, y_pred):
        u = ((y_pred - y_test) ** 2).sum()
        v = ((y_pred - y_test.mean()) ** 2).sum()
        r2 = (1 - u / v)
        return r2
#-----Learning on dataset with linear relation with semi-randomised noise
models = []

for NUM in range(0,2*MAX_NUM+1):
    AAA = (NUM-MAX_NUM)/X_DIV
   
    X_train = []
    y_train = []
    for n in range(1, NUM_TRAIN_SETS+2):
        np.random.seed(n)
        X_train.append(X_SIZE * np.random.rand(SIZE_PARAM, 1))
        y_train.append(AAA * 1 * X_train[-1] + np.random.normal(0, n*DISPERSION_PARAM, size=(SIZE_PARAM, 1)))

    X_train_all = np.concatenate(X_train)
    y_train_all = np.concatenate(y_train)

    model = CustomLinearRegression()
    model.fit(X_train, y_train)
    models.append(model)
    
    print(f"Learning: {int(NUM*100/(2*MAX_NUM+2))} %")
#-----Fitting external dataset 
INPUT_DATA = INPUT_DF
data = pd.read_csv(INPUT_DATA)
X_test = data['x1'].values.reshape(-1, 1)
y_test = data['y1'].values.reshape(-1, 1)

R2 = []
n_val  = []
coeff = []
n = 0


for M in models:
    n += 1
    y_pred = M.predict(X_test)
    r2 = M.score(y_test, y_pred)
    
    globals()[f"r{n}"] = r2
    R2.append(globals()[f"r{n}"])
        
    globals()[f"n{n}"] = n
    n_val.append(globals()[f"n{n}"])
        
    globals()[f"coeff{n}"] = M.coef_
    coeff.append(globals()[f"coeff{n}"])

    print(f"Fitting coefficients: {int(n*100/(2*MAX_NUM+2))} %")


CSV_path = "R2values.csv"
columns = ["n", "R2", "model"]
pd.DataFrame(columns=columns).to_csv(CSV_path, index=False)
    
df = pd.read_csv("R2values.csv")
df["n"] = n_val
df["R2"] = R2
df["model"] = coeff
df.to_csv("R2values.csv", index=False)

MODELcoef = df["model"][df["R2"].idxmax()]
r2max = df["R2"][df["R2"].idxmax()]
#------Outputting values of the fit
print("Coefficient of determination (R^2 score):", r2max)
print("Linear coefficient of fit function:", MODELcoef)

x = np.linspace(0, X_SIZE/X_DIV ,SIZE_PARAM)
y = x*MODELcoef
plt.plot(x, y, color='red', linewidth=2, label='Regression Line')
plt.scatter(X_test, y_test, color='blue', label='Test Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Test Data with Regression Line')
plt.legend()
plt.show()
