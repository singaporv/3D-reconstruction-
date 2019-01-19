'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import cv2 as cv 
import submission as sub
import helper
import random
import findM2


def find(M2s, C1, pts1, pts2, K2):
	N = M2s.shape[2]
	C2s = []
	for i in range(N):
		M2 = M2s[:,:,i]
		C2s.append(np.matmul(K2, M2s[:,:,i]))
		P, err = sub.triangulate(C1, pts1, C2s[i], pts2)
		if (P[:,-1] >= 0.0).all():
			break
	return M2, C2s[i], P