from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from requirements import *

def ESS(y_pred):
    return np.sum((y_pred-np.mean(y_pred))**2)

def RSS(y_true, y_pred):
    return np.sum((y_true-y_pred)**2)

def TSS(y_true):
    return np.sum((y_true-np.mean(y_true))**2)

def R_square(y_true, y_pred):
    return ESS(y_pred=y_pred)/TSS(y_true=y_true)

def R_square_adjusted(y_true, y_pred, number_of_independent_variables_and_const):
    return 1-(len(y_true)-1)*(1-R_square(y_true=y_true, y_pred=y_pred))/(len(y_true)-number_of_independent_variables_and_const)