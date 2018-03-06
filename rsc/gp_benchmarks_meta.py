
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import requests
import zipfile
import io
from math import sqrt, sin, cos, log, pi, e

import os

# In[]:

# Set the current dir to the dir where this script is located
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# In[81]:


def synthetic_gen(rnd, function, training_gen, test_gen=None):
    training_set = []
    test_set = []
    for i in range(training_gen.n):
        inst = training_gen.generate(rnd)
        training_set.append(inst + [function(*inst)])
    if test_gen:
        for i in range(test_gen.n):
            inst = test_gen.generate(rnd)
            test_set.append(inst + [function(*inst)])
    else:
        test_set = training_set
    return [training_set, test_set]

class U:
    def __init__(self, ini, end, n):
        self.ini = ini
        self.end = end
        self.n = n
    
    def generate(self, rnd):
        return [rnd.uniform(ini, end) for ini, end in zip(self.ini, self.end)]

class E:
    def __init__(self, ini, end, step):
        self.ini = ini
        self.end = end
        self.step = step
        
        mesh = np.meshgrid(*[np.arange(ini, end+step, step) 
                           for ini, end, step in zip(self.ini, self.end, self.step)])
        self.points = [dim.reshape(1,-1)[0] for dim in mesh]
        self.index = 0
        self.n = len(self.points[0])
    
    def generate(self, rnd):
        inst = [self.points[i][self.index] for i in range(len(self.points))]
        self.index += 1
        return inst
    
def p_log(x):
    if x == 0:
        return 0
    else:
        return log(abs(x))
    
def p_sqrt(x):
    return sqrt(abs(x))

def get_data(url, rnd=None, pd_sep=',', pd_header=None, pd_skiprows=None, dataset=None):
    if dataset == "BOH":
        from sklearn.datasets import load_boston
        boston = load_boston()
        df = pd.DataFrame(boston['data'])
        df = pd.concat([df, pd.Series(boston['target'])], axis=1)
    elif dataset == "CCP":
        # Get the file object from an url
        r = requests.get(url)
        # Create a ZipFile object from it
        z = zipfile.ZipFile(io.BytesIO(r.content))
        # Read from a xlsx file inside the zip file
        df = pd.read_excel(z.open('CCPP/Folds5x2_pp.xlsx'))
    elif dataset == "CST":
        df = pd.read_excel(url)
    elif dataset == "ENC":
        df = pd.read_excel(url)
        # Drop Y1
        df.drop("Y1", axis=1, inplace=True)
    elif dataset == "ENH":
        df = pd.read_excel(url)
        # Drop Y2
        df.drop("Y2", axis=1, inplace=True)
    else:
        df = pd.read_csv(url, header=pd_header, sep=pd_sep, skiprows=pd_skiprows)
        if dataset == "ABA":
            # Get dummy variables for the first column
            df_dummies = pd.get_dummies(df.iloc[:,0])
            # Drop the first column
            df.drop(df.columns[0], axis=1, inplace=True)
            # Concatenate the dummy variables with the data
            df = pd.concat([df_dummies, df], axis=1)
            df = df.sample(500, random_state=rnd.randrange(100), axis=0)
        elif dataset == "CPU":
            # Drop the first two columns
            df.drop(df.columns[[0,1]], axis=1, inplace=True)
        elif dataset == "FFR":
            df.drop(["month", "day"], axis=1, inplace=True)
        elif dataset == "OZO":
            # Imputation (replance NaN's by the mean of the column)
            df.fillna(df.mean(), inplace=True)
    return df
         
    
# In[82]:


seed = 1234
rnd = random.Random(seed)

data = {"Meier-3": synthetic_gen(rnd, 
                                 lambda x_1,x_2: (x_1**2*x_2**2)/(x_1+x_2), 
                                 U([-1, -1], [1, 1], 50), U([-1, -1], [1, 1], 50)),
        "Meier-4": synthetic_gen(rnd, 
                                 lambda x_1,x_2: x_1**5/x_2**3, 
                                 U([-1, -1], [1, 1], 50), U([-1, -1], [1, 1], 50)),
        "Nonic": synthetic_gen(rnd,
                               lambda x_1: sum([x_1**i for i in range(1,10)]), 
                               E([-1], [1], [2/19]), U([-1], [1], 20)),
        "Sine": synthetic_gen(rnd,
                              lambda x_1: sin(x_1), 
                              E([0], [6.2], [0.1])),
        "Burks": synthetic_gen(rnd,
                               lambda x_1: 4*x_1**4 + 3*x_1**3 + 2*x_1**2 + x_1, 
                               U([-1], [1], 20)),
        "R1": synthetic_gen(rnd,
                            lambda x_1: (x_1+1)**3/(x_1**2-x_1+1), 
                            E([-1], [1], [2/19]), U([-1], [1], 20)),
        "R2": synthetic_gen(rnd,
                            lambda x_1: (x_1**5-3*x_1**3+1)/(x_1**2+1), 
                            E([-1], [1], [2/19]), U([-1], [1], 20)),
        "R3": synthetic_gen(rnd,
                            lambda x_1: (x_1**6+x_1**5)/(x_1**4+x_1**3+x_1**2+x_1+1), 
                            E([-1], [1], [2/19]), U([-1], [1], 20)),
        "Poly-10": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10: 
                                     x_1*x_2+x_3*x_4+x_5*x_6+x_1*x_7*x_9+x_3*x_6*x_10,
                                 U([0]*10, [1]*10, 330), U([0]*10, [1]*10, 170)),
        "Koza-2": synthetic_gen(rnd,
                                lambda x_1: x_1**5-2*x_1**3+x_1, 
                                U([-1], [1], 20), U([-1], [1], 20)),
        "Koza-3": synthetic_gen(rnd,
                                lambda x_1: x_1**6-2*x_1**4+x_1**2,
                                U([-1], [1], 20), U([-1], [1], 20)),
        "Korns-1": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 1.57+24.3*x_4,
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-2": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 0.23+14.2*(x_4+x_2)/(3*x_5),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-3": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 
                                     -5.41+4.9*(x_4-x_1+x_2/x_5)/(3*x_5),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-4": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 
                                     -2.3+0.13*sin(x_3),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-5": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 3+2.13*p_log(x_5),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-6": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 1.3+0.13*p_sqrt(x_1),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-7": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 
                                     213.80940889*(1-e**(-0.54723748542*x_1)),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-8": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 
                                     6.87+11*p_sqrt(7.23*x_1*x_4*x_5),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-9": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 
                                     (p_sqrt(x_1)/p_log(x_2))*(e**(x_3)/x_4**2),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-10": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 
                                     0.81+24.3*(2*x_2+3*x_3**2)/(4*x_4**3+5*x_5**4),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-11": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 
                                     6.87+11*cos(7.23*x_1**3),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Korns-12": synthetic_gen(rnd,
                                 lambda x_1, x_2, x_3, x_4, x_5: 
                                     2-2.1*cos(9.8*x_1)*sin(1.3*x_5),
                                 U([-50]*5, [50]*5, 1000), U([-50]*5, [50]*5, 1000)),
        "Vladislavleva-1": synthetic_gen(rnd,
                                         lambda x_1, x_2: 
                                             e**(-(x_1-1)**2)/(1.2+(x_2-2.5)**2),
                                         U([0.3]*2, [4]*2, 100), E([-0.2]*2, [4.2]*2, [0.1]*2)),
        "Vladislavleva-2": synthetic_gen(rnd,
                                         lambda x_1: 
                                             e**(-x_1)*x_1**3*(cos(x_1)*sin(x_1))*(cos(x_1)*sin(x_1)**2-1),
                                         E([0.05], [10], [0.1]), E([-0.5], [10.5], [0.05])),
        "Vladislavleva-3": synthetic_gen(rnd,
                                         lambda x_1, x_2: 
                                             e**(-x_1)*x_1**3*(cos(x_1)*sin(x_1))*(cos(x_1)*sin(x_1)**2-1)*(x_2-5),
                                         E([0.05]*2, [10, 10.05], [0.1, 2]), E([-0.5]*2, [10.5]*2, [0.05, 0.5])),
        "Vladislavleva-4": synthetic_gen(rnd,
                                         lambda x_1, x_2, x_3, x_4, x_5: 
                                             10/(5+(x_1-3)**2+(x_2-3)**2+(x_3-3)**2+(x_4-3)**2+(x_5-3)**2),
                                         U([0.05]*5, [6.05]*5, 1024), U([-0.25]*5, [6.35]*5, 5000)),
        "Vladislavleva-5": synthetic_gen(rnd,
                                         lambda x_1, x_2, x_3: 
                                             30*(x_1-1)*(x_3-1)/((x_1-10)*x_2**2),
                                         U([0.05, 1, 0.05], [2]*3, 300), 
                                         E([-0.05, 0.95, -0.05], [2.1, 2.05, 2.1], [0.15, 0.1, 0.15])),
        "Vladislavleva-6": synthetic_gen(rnd,
                                         lambda x_1, x_2: 6*sin(x_1)*cos(x_2),
                                         U([0.1]*2, [5.9]*5, 30), 
                                         E([-0.05]*2, [6.05]*2, [0.02]*2)),
        "Vladislavleva-7": synthetic_gen(rnd,
                                         lambda x_1, x_2: 
                                             (x_1-3)*(x_2-3)+2*sin((x_1-4)*(x_2-4)),
                                         U([0.05]*2, [6.05]*2, 300), U([-0.25]*2, [6.35]*2, 1000)),
        "Vladislavleva-8": synthetic_gen(rnd,
                                         lambda x_1, x_2: 
                                             ((x_1-3)**4+(x_2-3)**3-(x_2-3))/((x_2-2)**4+10),
                                         U([0.05]*2, [6.05]*2, 50), E([-0.25]*2, [6.35]*2, [0.2]*2)),
        "Pagie-1": synthetic_gen(rnd,
                                 lambda x_1, x_2: 1/(1+x_1**(-4))+1/(1+x_2**(-4)),
                                 E([-5]*2, [5]*2, [0.4]*2)),
        "Keijzer-1": synthetic_gen(rnd,
                                   lambda x_1: 
                                       0.3*x_1*sin(2*pi*x_1),
                                   E([-1], [1], [0.1]),
                                   E([-1], [1], [0.001])),
        "Keijzer-2": synthetic_gen(rnd,
                                   lambda x_1: 
                                       0.3*x_1*sin(2*pi*x_1),
                                   E([-2], [2], [0.1]),
                                   E([-2], [2], [0.001])),
        "Keijzer-3": synthetic_gen(rnd,
                                   lambda x_1: 
                                       0.3*x_1*sin(2*pi*x_1),
                                   E([-3], [3], [0.1]),
                                   E([-3], [3], [0.001])),
        "Keijzer-4": synthetic_gen(rnd,
                                   lambda x_1: 
                                       x_1**3*e**(-x_1)*cos(x_1)*sin(x_1)*(sin(x_1)**2*cos(x_1)-1),
                                   E([0], [10], [0.05]),
                                   E([0.05], [10.05], [0.05])),
        "Keijzer-5": synthetic_gen(rnd,
                                   lambda x_1, x_2, x_3: 30*x_1*x_3/((x_1-10)*x_2**2),
                                   U([-1, 1, -1], [1,2,1], 1000),
                                   U([-1, 1, -1], [1,2,1], 10000)),
        "Keijzer-6": synthetic_gen(rnd,
                                   lambda x_1: sum([1/i for i in range(1, x_1+1)]),
                                   E([1], [50], [1]),
                                   E([1], [120], [1])),
        "Keijzer-7": synthetic_gen(rnd,
                                   lambda x_1: log(x_1),
                                   E([1], [100], [1]),
                                   E([1], [100], [0.1])),
        "Keijzer-8": synthetic_gen(rnd,
                                   lambda x_1: sqrt(x_1),
                                   E([0], [100], [1]),
                                   E([0], [100], [0.1])),
        "Keijzer-9": synthetic_gen(rnd,
                                   lambda x_1: log(x_1+sqrt(x_1**2+1)),
                                   E([0], [100], [1]),
                                   E([0], [100], [0.1])),
        "Keijzer-10": synthetic_gen(rnd,
                                   lambda x_1, x_2: x_1**x_2,
                                   U([0]*2, [1]*2, 100),
                                   E([0]*2, [1]*2, [0.01]*2)),
        "Keijzer-11": synthetic_gen(rnd,
                                   lambda x_1, x_2: x_1*x_2+sin((x_1-1)*(x_2-1)),
                                   U([-3]*2, [3]*2, 20),
                                   E([-3]*2, [3]*2, [0.01]*2)),
        "Keijzer-12": synthetic_gen(rnd,
                                   lambda x_1, x_2: x_1**4-x_1**3+(x_2**2/2)-x_2,
                                   U([-3]*2, [3]*2, 20),
                                   E([-3]*2, [3]*2, [0.01]*2)),
        "Keijzer-13": synthetic_gen(rnd,
                                   lambda x_1, x_2: 6*sin(x_1)*cos(x_2),
                                   U([-3]*2, [3]*2, 20),
                                   E([-3]*2, [3]*2, [0.01]*2)),
        "Keijzer-14": synthetic_gen(rnd,
                                   lambda x_1, x_2: 8/(2+x_1**2+x_2**2),
                                   U([-3]*2, [3]*2, 20),
                                   E([-3]*2, [3]*2, [0.01]*2)),
        "Keijzer-15": synthetic_gen(rnd,
                                   lambda x_1, x_2: (x_1**3/5)+(x_2**3/2)-x_2-x_1,
                                   U([-3]*2, [3]*2, 20),
                                   E([-3]*2, [3]*2, [0.01]*2)),
        "Nguyen-1": synthetic_gen(rnd,
                                   lambda x_1: x_1**3+x_1**2+x_1,
                                   U([-1], [1], 20),
                                   U([-1], [1], 20)),
        "Nguyen-2": synthetic_gen(rnd,
                                   lambda x_1: x_1**4+x_1**3+x_1**2+x_1,
                                   U([-1], [1], 20),
                                   U([-1], [1], 20)),
        "Nguyen-3": synthetic_gen(rnd,
                                   lambda x_1: x_1**5+x_1**4+x_1**3+x_1**2+x_1,
                                   U([-1], [1], 20),
                                   U([-1], [1], 20)),
        "Nguyen-4": synthetic_gen(rnd,
                                   lambda x_1: x_1**6+x_1**5+x_1**4+x_1**3+x_1**2+x_1,
                                   U([-1], [1], 20),
                                   U([-1], [1], 20)),
        "Nguyen-5": synthetic_gen(rnd,
                                   lambda x_1: sin(x_1**2)*cos(x_1)-1,
                                   U([-1], [1], 20),
                                   U([-1], [1], 20)),
        "Nguyen-6": synthetic_gen(rnd,
                                   lambda x_1: sin(x_1)+sin(x_1+x_1**2),
                                   U([-1], [1], 20),
                                   U([-1], [1], 20)),
        "Nguyen-7": synthetic_gen(rnd,
                                   lambda x_1: log(x_1+1)+log(x_1**2+1),
                                   U([0], [2], 20),
                                   U([0], [2], 20)),
        "Nguyen-8": synthetic_gen(rnd,
                                   lambda x_1: sqrt(x_1),
                                   U([0], [4], 20),
                                   U([0], [4], 20)),
        "Nguyen-9": synthetic_gen(rnd,
                                   lambda x_1, x_2: sin(x_1)+sin(x_2**2),
                                   U([-1]*2, [1]*2, 100),
                                   U([-1]*2, [1]*2, 100)),
        "Nguyen-10": synthetic_gen(rnd,
                                   lambda x_1, x_2: 2*sin(x_1)*cos(x_2),
                                   U([-1]*2, [1]*2, 100),
                                   U([-1]*2, [1]*2, 100)),
        "Abalone": get_data('https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',
                            rnd=rnd, dataset='ABA'),
        "Airfoil": get_data('https://archive.ics.uci.edu/ml/'
                            'machine-learning-databases/00291/airfoil_self_noise.dat',
                            pd_sep="\t"),
        "Boston": get_data("", dataset="BOH"),
        "Combined-cycle": get_data('https://archive.ics.uci.edu/ml/'
                                   'machine-learning-databases/00294/CCPP.zip',
                                   dataset="CCP"),
        "Computer-hardware": get_data('https://archive.ics.uci.edu/ml/'
                                      'machine-learning-databases/cpu-performance/machine.data',
                                      dataset="CPU"),
        "Concrete-strength": get_data('https://archive.ics.uci.edu/ml/'
                                      'machine-learning-databases/concrete/compressive/Concrete_Data.xls',
                                      dataset="CST"),
        "Energy-cooling": get_data('http://archive.ics.uci.edu/ml/'
                                   'machine-learning-databases/00242/ENB2012_data.xlsx',
                                   dataset="ENC"),
        "Energy-heating": get_data('http://archive.ics.uci.edu/ml/'
                                   'machine-learning-databases/00242/ENB2012_data.xlsx',
                                   dataset="ENH"),
        "Forest-fire": get_data('https://archive.ics.uci.edu/ml/'
                                'machine-learning-databases/forest-fires/forestfires.csv',
                                pd_header='infer', dataset="FFR"),
        "Ozone": get_data('./data/ozone.data', dataset="OZO"),
        "Wine-quality-red": get_data('https://archive.ics.uci.edu/ml/'
                                     'machine-learning-databases/wine-quality/winequality-red.csv',
                                     pd_header='infer', pd_sep=';'),
        "Wine-quality-white": get_data('https://archive.ics.uci.edu/ml/'
                                       'machine-learning-databases/wine-quality/winequality-white.csv', 
                                       pd_header='infer', pd_sep=';'),
        "Yacht": get_data('./data/yacht.data', pd_sep='\s+')
                                       }
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       
                                       