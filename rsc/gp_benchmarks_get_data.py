import pandas as pd
import numpy as np
import requests
import zipfile
import io
from math import sqrt, sin, cos, log, pi, e
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from sklearn.model_selection import KFold
from multiprocessing import Pool, cpu_count

import os



# Set the current dir to the dir where this script is located
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# In[81]:


def synthetic_gen(name, rnd, function, training_gen, test_gen=None):
    print("Generating the \"" + name + "\" dataset")
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
    return {"training": pd.DataFrame(training_set),
            "test": pd.DataFrame(test_set)}

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
    
def get_data(name, url, rnd=None, pd_sep=',', pd_header=None, pd_skiprows=None, dataset=None):
    print("Reading the \"" + name + "\" dataset")
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
            df = df.sample(500, random_state=rnd, axis=0)
        elif dataset == "CPU":
            # Drop the first two columns
            df.drop(df.columns[[0,1]], axis=1, inplace=True)
        elif dataset == "FFR":
            df.drop(["month", "day"], axis=1, inplace=True)
        elif dataset == "OZO":
            # Imputation (replance NaN's by the mean of the column)
            df.fillna(df.mean(), inplace=True)
    return df.apply(np.float64)
         
    
# In[82]:


seed = 1234
rnd = np.random.RandomState(seed)

data_synt = {"Meier-3": synthetic_gen("Meier-3", rnd, 
                                      lambda x_1,x_2: (x_1**2*x_2**2)/(x_1+x_2), 
                                      U([-1, -1], [1, 1], 50), U([-1, -1], [1, 1], 50)),
            "Meier-4": synthetic_gen("Meier-4", rnd, 
                                     lambda x_1,x_2: x_1**5/x_2**3, 
                                     U([-1, -1], [1, 1], 50), U([-1, -1], [1, 1], 50)),
            "Nonic": synthetic_gen("Nonic", rnd,
                                   lambda x_1: sum([x_1**i for i in range(1,10)]), 
                                   E([-1], [1], [2/19]), U([-1], [1], 20)),
            "Sine": synthetic_gen("Sine", rnd,
                                  lambda x_1: sin(x_1), 
                                  E([0], [6.2], [0.1])),
            "Burks": synthetic_gen("Burks", rnd,
                                   lambda x_1: 4*x_1**4 + 3*x_1**3 + 2*x_1**2 + x_1, 
                                   U([-1], [1], 20)),
            "R1": synthetic_gen("R1", rnd,
                                lambda x_1: (x_1+1)**3/(x_1**2-x_1+1), 
                                E([-1], [1], [2/19]), U([-1], [1], 20)),
            "R2": synthetic_gen("R2", rnd,
                                lambda x_1: (x_1**5-3*x_1**3+1)/(x_1**2+1), 
                                E([-1], [1], [2/19]), U([-1], [1], 20)),
            "R3": synthetic_gen("R3", rnd,
                                lambda x_1: (x_1**6+x_1**5)/(x_1**4+x_1**3+x_1**2+x_1+1), 
                                E([-1], [1], [2/19]), U([-1], [1], 20)),
            "Poly-10": synthetic_gen("Poly-10", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10: 
                                         x_1*x_2+x_3*x_4+x_5*x_6+x_1*x_7*x_9+x_3*x_6*x_10,
                                     U([0]*10, [1]*10, 330), U([0]*10, [1]*10, 170)),
            "Koza-2": synthetic_gen("Koza-2", rnd,
                                    lambda x_1: x_1**5-2*x_1**3+x_1, 
                                    U([-1], [1], 20), U([-1], [1], 20)),
            "Koza-3": synthetic_gen("Koza-3", rnd,
                                    lambda x_1: x_1**6-2*x_1**4+x_1**2,
                                    U([-1], [1], 20), U([-1], [1], 20)),
            "Korns-1": synthetic_gen("Korns-1", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 1.57+24.3*x_4,
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "Korns-4": synthetic_gen("Korns-4", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 
                                         -2.3+0.13*sin(x_3),
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "Korns-7": synthetic_gen("Korns-7", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 
                                         213.80940889*(1-e**(-0.54723748542*x_1)),
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "Korns-11": synthetic_gen("Korns-11", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 
                                         6.87+11*cos(7.23*x_1**3),
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "Korns-12": synthetic_gen("Korns-12", rnd,
                                     lambda x_1, x_2, x_3, x_4, x_5: 
                                         2-2.1*cos(9.8*x_1)*sin(1.3*x_5),
                                     U([-50]*5, [50]*5, 10000), U([-50]*5, [50]*5, 10000)),
            "Vladislavleva-1": synthetic_gen("Vladislavleva-1", rnd,
                                             lambda x_1, x_2: 
                                                 e**(-(x_1-1)**2)/(1.2+(x_2-2.5)**2),
                                             U([0.3]*2, [4]*2, 100), E([-0.2]*2, [4.2]*2, [0.1]*2)),
            "Vladislavleva-2": synthetic_gen("Vladislavleva-2", rnd,
                                             lambda x_1: 
                                                 e**(-x_1)*x_1**3*(cos(x_1)*sin(x_1))*(cos(x_1)*sin(x_1)**2-1),
                                             E([0.05], [10], [0.1]), E([-0.5], [10.5], [0.05])),
            "Vladislavleva-3": synthetic_gen("Vladislavleva-3", rnd,
                                             lambda x_1, x_2: 
                                                 e**(-x_1)*x_1**3*(cos(x_1)*sin(x_1))*(cos(x_1)*sin(x_1)**2-1)*(x_2-5),
                                             E([0.05]*2, [10, 10.05], [0.1, 2]), E([-0.5]*2, [10.5]*2, [0.05, 0.5])),
            "Vladislavleva-4": synthetic_gen("Vladislavleva-4", rnd,
                                             lambda x_1, x_2, x_3, x_4, x_5: 
                                                 10/(5+(x_1-3)**2+(x_2-3)**2+(x_3-3)**2+(x_4-3)**2+(x_5-3)**2),
                                             U([0.05]*5, [6.05]*5, 1024), U([-0.25]*5, [6.35]*5, 5000)),
            "Vladislavleva-5": synthetic_gen("Vladislavleva-5", rnd,
                                             lambda x_1, x_2, x_3: 
                                                 30*(x_1-1)*(x_3-1)/((x_1-10)*x_2**2),
                                             U([0.05, 1, 0.05], [2]*3, 300), 
                                             E([-0.05, 0.95, -0.05], [2.1, 2.05, 2.1], [0.15, 0.1, 0.15])),
            "Vladislavleva-6": synthetic_gen("Vladislavleva-6", rnd,
                                             lambda x_1, x_2: 6*sin(x_1)*cos(x_2),
                                             U([0.1]*2, [5.9]*5, 30), 
                                             E([-0.05]*2, [6.05]*2, [0.02]*2)),
            "Vladislavleva-7": synthetic_gen("Vladislavleva-7", rnd,
                                             lambda x_1, x_2: 
                                                 (x_1-3)*(x_2-3)+2*sin((x_1-4)*(x_2-4)),
                                             U([0.05]*2, [6.05]*2, 300), U([-0.25]*2, [6.35]*2, 1000)),
            "Vladislavleva-8": synthetic_gen("Vladislavleva-8", rnd,
                                             lambda x_1, x_2: 
                                                 ((x_1-3)**4+(x_2-3)**3-(x_2-3))/((x_2-2)**4+10),
                                             U([0.05]*2, [6.05]*2, 50), E([-0.25]*2, [6.35]*2, [0.2]*2)),
            "Pagie-1": synthetic_gen("Pagie-1", rnd,
                                     lambda x_1, x_2: 1/(1+x_1**(-4))+1/(1+x_2**(-4)),
                                     E([-5]*2, [5]*2, [0.4]*2)),
            "Keijzer-1": synthetic_gen("Keijzer-1", rnd,
                                       lambda x_1: 
                                           0.3*x_1*sin(2*pi*x_1),
                                       E([-1], [1], [0.1]),
                                       E([-1], [1], [0.001])),
            "Keijzer-2": synthetic_gen("Keijzer-2", rnd,
                                       lambda x_1: 
                                           0.3*x_1*sin(2*pi*x_1),
                                       E([-2], [2], [0.1]),
                                       E([-2], [2], [0.001])),
            "Keijzer-3": synthetic_gen("Keijzer-3", rnd,
                                       lambda x_1: 
                                           0.3*x_1*sin(2*pi*x_1),
                                       E([-3], [3], [0.1]),
                                       E([-3], [3], [0.001])),
            "Keijzer-4": synthetic_gen("Keijzer-4", rnd,
                                       lambda x_1: 
                                           x_1**3*e**(-x_1)*cos(x_1)*sin(x_1)*(sin(x_1)**2*cos(x_1)-1),
                                       E([0], [10], [0.05]),
                                       E([0.05], [10.05], [0.05])),
            "Keijzer-5": synthetic_gen("Keijzer-5", rnd,
                                       lambda x_1, x_2, x_3: 30*x_1*x_3/((x_1-10)*x_2**2),
                                       U([-1, 1, -1], [1,2,1], 1000),
                                       U([-1, 1, -1], [1,2,1], 10000)),
            "Keijzer-6": synthetic_gen("Keijzer-6", rnd,
                                       lambda x_1: sum([1/i for i in range(1, x_1+1)]),
                                       E([1], [50], [1]),
                                       E([1], [120], [1])),
            "Keijzer-7": synthetic_gen("Keijzer-7", rnd,
                                       lambda x_1: log(x_1),
                                       E([1], [100], [1]),
                                       E([1], [100], [0.1])),
            "Keijzer-8": synthetic_gen("Keijzer-8", rnd,
                                       lambda x_1: sqrt(x_1),
                                       E([0], [100], [1]),
                                       E([0], [100], [0.1])),
            "Keijzer-9": synthetic_gen("Keijzer-9", rnd,
                                       lambda x_1: log(x_1+sqrt(x_1**2+1)),
                                       E([0], [100], [1]),
                                       E([0], [100], [0.1])),
            "Keijzer-10": synthetic_gen("Keijzer-10", rnd,
                                       lambda x_1, x_2: x_1**x_2,
                                       U([0]*2, [1]*2, 100),
                                       E([0]*2, [1]*2, [0.01]*2)),
            "Keijzer-11": synthetic_gen("Keijzer-11", rnd,
                                       lambda x_1, x_2: x_1*x_2+sin((x_1-1)*(x_2-1)),
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "Keijzer-12": synthetic_gen("Keijzer-12", rnd,
                                       lambda x_1, x_2: x_1**4-x_1**3+(x_2**2/2)-x_2,
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "Keijzer-13": synthetic_gen("Keijzer-13", rnd,
                                       lambda x_1, x_2: 6*sin(x_1)*cos(x_2),
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "Keijzer-14": synthetic_gen("Keijzer-14", rnd,
                                       lambda x_1, x_2: 8/(2+x_1**2+x_2**2),
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "Keijzer-15": synthetic_gen("Keijzer-15", rnd,
                                       lambda x_1, x_2: (x_1**3/5)+(x_2**3/2)-x_2-x_1,
                                       U([-3]*2, [3]*2, 20),
                                       E([-3]*2, [3]*2, [0.01]*2)),
            "Nguyen-1": synthetic_gen("Nguyen-1", rnd,
                                       lambda x_1: x_1**3+x_1**2+x_1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "Nguyen-2": synthetic_gen("Nguyen-2", rnd,
                                       lambda x_1: x_1**4+x_1**3+x_1**2+x_1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "Nguyen-3": synthetic_gen("Nguyen-3", rnd,
                                       lambda x_1: x_1**5+x_1**4+x_1**3+x_1**2+x_1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "Nguyen-4": synthetic_gen("Nguyen-4", rnd,
                                       lambda x_1: x_1**6+x_1**5+x_1**4+x_1**3+x_1**2+x_1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "Nguyen-5": synthetic_gen("Nguyen-5", rnd,
                                       lambda x_1: sin(x_1**2)*cos(x_1)-1,
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "Nguyen-6": synthetic_gen("Nguyen-6", rnd,
                                       lambda x_1: sin(x_1)+sin(x_1+x_1**2),
                                       U([-1], [1], 20),
                                       U([-1], [1], 20)),
            "Nguyen-7": synthetic_gen("Nguyen-7", rnd,
                                       lambda x_1: log(x_1+1)+log(x_1**2+1),
                                       U([0], [2], 20),
                                       U([0], [2], 20)),
            "Nguyen-8": synthetic_gen("Nguyen-8", rnd,
                                       lambda x_1: sqrt(x_1),
                                       U([0], [4], 20),
                                       U([0], [4], 20)),
            "Nguyen-9": synthetic_gen("Nguyen-9", rnd,
                                       lambda x_1, x_2: sin(x_1)+sin(x_2**2),
                                       U([-1]*2, [1]*2, 100),
                                       U([-1]*2, [1]*2, 100)),
            "Nguyen-10": synthetic_gen("Nguyen-10", rnd,
                                       lambda x_1, x_2: 2*sin(x_1)*cos(x_2),
                                       U([-1]*2, [1]*2, 100),
                                       U([-1]*2, [1]*2, 100))
            }
# In[ ]:    
    
seed = 4567
rnd = np.random.RandomState(seed)
    
data_real = {"Abalone": get_data('Abalone', 
                                 'https://archive.ics.uci.edu/ml/'
                                 'machine-learning-databases/abalone/abalone.data',
                                 rnd=rnd, dataset='ABA'),
             "Airfoil": get_data('Airfoil',
                                 'https://archive.ics.uci.edu/ml/'
                                 'machine-learning-databases/00291/airfoil_self_noise.dat',
                                 pd_sep="\t"),
            "Boston": get_data('Boston', "", dataset="BOH"),
            "Combined-cycle": get_data('Combined-cycle', 
                                       'https://archive.ics.uci.edu/ml/'
                                       'machine-learning-databases/00294/CCPP.zip',
                                       dataset="CCP"),
            "Computer-hardware": get_data('Computer-hardware', 
                                          'https://archive.ics.uci.edu/ml/'
                                          'machine-learning-databases/cpu-performance/machine.data',
                                          dataset="CPU"),
            "Concrete-strength": get_data('Concrete-strength', 
                                          'https://archive.ics.uci.edu/ml/'
                                          'machine-learning-databases/concrete/compressive/Concrete_Data.xls',
                                          dataset="CST"),
            "Energy-cooling": get_data('Energy-cooling', 
                                       'http://archive.ics.uci.edu/ml/'
                                       'machine-learning-databases/00242/ENB2012_data.xlsx',
                                       dataset="ENC"),
            "Energy-heating": get_data('Energy-heating', 
                                       'http://archive.ics.uci.edu/ml/'
                                       'machine-learning-databases/00242/ENB2012_data.xlsx',
                                       dataset="ENH"),
            "Forest-fire": get_data('Forest-fire', 
                                    'https://archive.ics.uci.edu/ml/'
                                    'machine-learning-databases/forest-fires/forestfires.csv',
                                    pd_header='infer', dataset="FFR"),
            "Ozone": get_data('Ozone', './data/ozone.data', dataset="OZO"),
            "Wine-quality-red": get_data('Wine-quality-red', 
                                         'https://archive.ics.uci.edu/ml/'
                                         'machine-learning-databases/wine-quality/winequality-red.csv',
                                         pd_header='infer', pd_sep=';'),
            "Wine-quality-white": get_data('Wine-quality-white', 
                                           'https://archive.ics.uci.edu/ml/'
                                           'machine-learning-databases/wine-quality/winequality-white.csv', 
                                           pd_header='infer', pd_sep=';'),
            "Yacht": get_data('Yacht', './data/yacht.data', pd_sep='\s+')
            }
                                       
                                       
# In[]:

def _aq(x1, x2):
    return np.divide(x1, np.sqrt(1 + x2**2))   

aq = make_function(function=_aq,
                   name='aq',
                   arity=2)

def evaluate(dataset_lst, n_jobs=None, n_rep=30, cv=None):
    stats = {}
    experiments = []
    seed = 1234
    rnd = np.random.RandomState(seed)
    kf = KFold(n_splits=5, shuffle=True, random_state=rnd)
    
    for data_name, data_points in dataset_lst.items():
        stats[data_name] = {}
        # Reset the pseudo-random number generator for each dataset (used by GP)
        rnd = np.random.RandomState(seed)
        
        if cv is not None: 
            partitions = list(kf.split(data_points))
        
        for i in range(n_rep):
            if cv is not None: 
                training = data_points.iloc[partitions[i % cv][0],:]
                test = data_points.iloc[partitions[i % cv][1],:]
            else:
                training = data_points['training']
                test = data_points['test']
            experiments.append([data_name, i+1, training, test, rnd.randint(10**6)])
    if not n_jobs or n_jobs > 1:    
        with Pool(processes=n_jobs) as pool:
            for exec_stats in pool.imap(worker, experiments):
                data_name = exec_stats[0]
                for stats_name, stats_value in exec_stats[1].items():
                    if stats_name not in stats[data_name]:
                        stats[data_name][stats_name] = []
                    stats[data_name][stats_name].append(stats_value)
    else:
        for experiment in experiments:
            exec_stats = worker(experiment)
            data_name = exec_stats[0]
            for stats_name, stats_value in exec_stats[1].items():
                if stats_name not in stats[data_name]:
                    stats[data_name][stats_name] = []
                stats[data_name][stats_name].append(stats_value)
    return stats

def worker(param):
    print("Dataset \"" + param[0] + "\", exec.", param[1])
    return [param[0], run_gp(training=param[2], test=param[3], rnd=param[4])]

def run_gp(training, test, rnd=None):
    # Normalization constant used to compute the normalized RMSE
    norm_const = np.sqrt(training.shape[0]/(training.shape[0]-1))/training.iloc[:,-1].std()
    
    # Genetic programming instance used to perform symbolic regression in the datasets
    function_set = ['add', 'sub', 'mul', aq, 'sqrt', 'sin']
    gp_param = {'population_size': 1000,
                  'generations': 50, 
                  'stopping_criteria': 0.00,
                  'const_range': (-1,1),
                  'init_depth': (2,6),
                  'init_method': 'half and half',
                  'metric': make_fitness(function=lambda y, y_pred, w:
                      np.sqrt(np.average(((y_pred - y) ** 2), weights=w)) * norm_const,
                                           greater_is_better=False),
                  'tournament_size': 10,
                  'p_crossover': 0.85, 
                  'function_set': function_set,
                  'p_subtree_mutation': 0.05,
                  'p_hoist_mutation': 0.05, 
                  'p_point_mutation': 0.05,
                  'max_samples': 1, 
                  'verbose': 0,
                  'parsimony_coefficient': 0.001, 
                  'random_state': rnd,
                  'n_jobs': 1}
    est_gp = SymbolicRegressor(**gp_param)
    est_gp.fit(training.iloc[:,:-1], training.iloc[:,-1])
    
    #norm_const = np.sqrt(training.shape[0]/(training.shape[0]-1))/training.iloc[:,-1].std()
    stats = {'size': est_gp._program.length_,
             'tr_rmse': est_gp._program.raw_fitness_ / norm_const,
             'tr_nrmse': est_gp._program.raw_fitness_}
    y_est = est_gp.predict(test.iloc[:,:-1])
    rmse = np.sqrt(np.average((y_est - test.iloc[:,-1]) ** 2))
    norm_const = np.sqrt(test.shape[0]/(test.shape[0]-1))/test.iloc[:,-1].std()
    
    stats['ts_rmse'] = rmse
    stats['ts_nrmse'] = rmse * norm_const
    
    return stats

def write_stats(stats_dict, path='./performace_metrics/'):
    """Write statistics returned by worker/run_gp methods.

    Extended description of function.

    Parameters
    ----------
    stats_dict : dict
        A dictionary containing results from different datasets. The format is
        in the form stats_dict['data_set_name']['statistics_name']
    arg2 : str
        Description of arg2

    Returns
    -------
    bool
        Description of return value
    """
    for key, data in stats_dict.items():
        df = pd.DataFrame(data)
        df.to_csv(path + key.lower() + '.csv')


# In[]:

stats_synt = evaluate(data_synt, n_rep = 30)
write_stats(stats_synt)

stats_real = evaluate(data_real, n_rep = 30, cv = 5)
write_stats(stats_real)
