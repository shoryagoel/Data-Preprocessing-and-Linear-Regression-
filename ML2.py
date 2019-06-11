import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Salary_Data.csv');

x = data['YearsExperience'].values
y = data['Salary'].values

def pred(x,y):
        m =  ((np.mean(x))*(np.mean(y))-np.mean(x*y))/(((np.mean(x))**2)-np.mean(x**2))
        b = np.mean(y) - m*np.mean(x)
        print(m,b)
        y_new=[]
        for i in x:
            y_new.append(b + m*i)
        return y_new;  
        

def Standard_error_line(y,y_new):
        sdl = sum((y-y_new)**2)
        return sdl;
        
def Standard_error_mean(y):
        y_mean=[]
        for i in y:
            y_mean.append(np.mean(y))
        sdm = sum((y-y_mean)**2)
        return sdm;
        

def cofficient(x,y):
        y_new = pred(x,y)
        sdl = Standard_error_line(y,y_new)
        sdm = Standard_error_mean(y)
        r = 1 - (sdl/sdm)
        return r;
        
r = cofficient(x,y)
print(r)

plt.scatter(x,y,color="green")
plt.plot(x,y_new,color="orange")
plt.show()