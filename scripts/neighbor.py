import numpy as np

# Check if neighbors are present, if they are return a list of indices, if not return 0
def check_neighbor(i, x, y, verbose=False):
    xi = x[i]
    yi = y[i]
    dx = x[0] - x[1]
    l = np.where(((x-xi)**2+(y-yi)**2)<(dx*1.01)**2)
    
    if verbose==True:
        print(" l : ", l,"\n")
    
    # 4 neighbors and the point itself!
    if((len(l[0])<5)):
        return 0
    else:
        return l