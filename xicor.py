#Python Xi correlation coefficient array calculation for both ties and no ties data

import numpy as np
from scipy.stats import rankdata, norm

def xicor(x, y, ties=False):

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    n = len(x)
    
    if len(y) != n:
        raise IndexError(f'X & Y variables array size mismatch: {len(x)}, {len(y)}')
    
    y = y[np.argsort(x)]
    r = rankdata(y, method='ordinal')
    nominator = np.sum(np.abs(np.diff(r)))
    
    if ties:
        
        l = rankdata(y, method='max')
        denominator = 2 * np.sum(l * (n-1))
        nominator *= n
        
    else:
        
        denominator = np.power(n, 2) - 1
        nominator *= 3  
        
    xi = 1 - nominator / denominator 
    p_value = norm.sf(xi, scale=2/5/np.sqrt(n))
    
    return xi, p_value
