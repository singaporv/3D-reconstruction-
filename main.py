import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import cv2 as cv 
import submission as sub
import helper
import random
import findM2
import scipy.ndimage.filters as fi
# import checkA4Format



if __name__ == '__main__':
	data = np.load('../data/some_corresp.npz')
	im1 = plt.imread('../data/im1.png')
	im2 = plt.imread('../data/im2.png')
	camera_cal = np.load('../data/intrinsics.npz') # Intrinsics 
	
	M = 640

	# for n in camera_cal.items():
	# 	print (n)

	N = len(data['pts1']) # N = 110

	# 8 point Algorithm
	# F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
	# helper.displayEpipolarF(im1, im2, F8)
	# print (F8)
	# np.savez('../results/q2_1.npz', F = F8, M = M)


	# Q2.2
	# 7 point Algorithm

	# Creating 7 random points from the sample points given
	#### np.random.seed(4) # For checking purpose only
	# rand = random.sample(range(0, N), 7)
	# pts1 = data['pts1']
	# pts2 = data ['pts2']
	# pts1_r = []
	# pts2_r = []
	# for r in rand:
	# 	pts1_r.append(pts1[r,:])
	# 	pts2_r.append(pts2[r,:])
	# pts1_r = np.asarray(pts1_r)
	# pts2_r = np.asarray(pts2_r)

	# Farray is a list array of length either 1 or 3 containing Fundamental matrix/matrices
	# Farray = sub.sevenpoint(pts1_r, pts2_r, M)
	# print (len(Farray))
	# print (Farray)
	# helper.displayEpipolarF(im1, im2, Farray[0])
	# helper.displayEpipolarF(im1, im2, Farray[1])
	# helper.displayEpipolarF(im1, im2, Farray[2])

	# F7 = Farray[0] 
	# np.savez('../results/q2_2.npz', F = Farray[0], M = M, pts1 = pts1_r, pts2 = pts2_r)


	# # Q3.1
	K1 = camera_cal['K1']
	K2 =  camera_cal['K2']
	E = sub.essentialMatrix(F, K1, K2)
	# E = sub.essentialMatrix(F8, K1, K2)
	# # print (E)


	# # # # Q3.2
	M1 = np.array([[1, 0, 0,0], [0, 1, 0,0],[0,0,1,0]])
	M2s = helper.camera2(E)

	C1 = np.matmul(K1, M1)
	C2s = []
	for j in range(4):
		temp = np.matmul(K2, M2s[:,:,j])
		C2s.append(temp)

	# # P, err = sub.triangulate(C1, data['pts1'], C2s[0], data['pts2'])
	# # print (err)

	# # Q3.3
	# M2, C2, P = findM2.find(M2s, C1, data['pts1'], data['pts2'], K2)
	# print (M2)
	# np.savez('../results/q3_3.npz', M2 = M2, C2 = C2, P = P)


	# Q4.1
	# np.savez('../results/q4_1.npz', F = F8, pts1 = data['pts1'], pts2 = data['pts2'])

	# helper.epipolarMatchGUI(im1, im2, F8)
	# Code edited in helper function for saving pts1 and pts2

	# Q4.2
	# See viualize.py


	# Q5.1
	data2 = np.load('../data/some_corresp_noisy.npz')
	# for n,v in data2.items():
	# 	print (n)
	F, inlier_1, inlier_2 = sub.ransacF(data2['pts1'], data2['pts2'], M)
	# helper.displayEpipolarF(im1, im2, F)


	# Computing F8 for noisy points
	# F8_noisy = sub.eightpoint(data2['pts1'], data2['pts2'], M)

	# helper.displayEpipolarF(im1, im2, F8_noisy)




	# Q5.2
	# a)
	# r = np.ones([3, 1])
	# R = sub.rodrigues(r)
	# print (R)

	# b)
	# R = np.array([[1,0,1],[0,1,0],[1,1,1]])
	# # R = np.array([[0.2, 0.38685218, 0.38685218], [0.38685218, 0.22629564, 0.38685218],[0.38685218, 0.38685218,0.22629564]])
	# print (R.shape)
	# r = sub.invRodrigues(R)
	# print (r.shape)
	# print (r)

	# Q5.3
	# E = sub.essentialMatrix(F, K1, K2)
	# M2s = helper.camera2(E)
	
	
	M1 = np.array([[1, 0, 0,0], [0, 1, 0,0],[0,0,1,0]])
	M2s = helper.camera2(E)

	C1 = np.matmul(K1, M1)
	C2s = []
	for j in range(4):
		temp = np.matmul(K2, M2s[:,:,j])
		C2s.append(temp)

	K1 = camera_cal['K1']
	K2 =  camera_cal['K2']
	E = sub.essentialMatrix(F, K1, K2)








	# Good reprojection error and plot
	M2, C2, P = findM2.find(M2s, C1, inlier_1, inlier_2, K2)

	P, err = sub.triangulate(C1, inlier_1, C2, inlier_2)
	print ('Error after bundle adjustment = ', err)
	# plot P


	# Bad reprojection error and plot
	M2, P = sub.bundleAdjustment(K1, M1, inlier_1, K2, M2, inlier_2, P)

	C2 = np.matmul(K2, M2)
	P, err = sub.triangulate(C1, inlier_1, C2, inlier_2)
	print ("error after bundle adjustment = ", err)
	# plot




	# Plotting
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')

	x = P[:,0]
	y = P[:,1]
	z = P[:,2]
	plt.gca().set_aspect('equal',adjustable = 'box')
	ax.scatter(x,y,z, color='blue')

	for i in range(P.shape[0]):
		ax.scatter(P[i][0], P[i][1], P[i][2], c='r', marker='o')

	ax.set_xlabel('X Axis')
	ax.set_ylabel('Y Axis')
	ax.set_zlabel('Z Axis')

	plt.show()
	plt.close()
