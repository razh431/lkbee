import numpy as np
import cv2


import pickle
import numpy as np

pickle_in = open("rachel_5G_data.pickle", "rb")
result = pickle.load(pickle_in)



def load():
    print(result)

def test():
    a = [[1,2,5],[6,9,0],[4,8,3]]
    a = np.array(a)
    length = len(a)
    pair = []

    while(len(np.where(a > -1)[0]) > 0):
        b = np.ravel(a)

        index = np.where(b >= 0, b, np.inf).argmin()
        print(index)
        tnum = index//len(a) #row
        dnum = index%len(a) #column
        #i gave up on filling it so here's iteration
        for x in range(len(a[tnum])):
            a[tnum][x] = -1
            a[x][dnum] = -1
        pair.append([tnum, dnum])
        print(pair)


# test()
load()
