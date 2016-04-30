import os
import csv
import numpy as np
import random

def hypothesis(th_0, th_1, inp_data):
	#print("calculating hypothesis")
	hypo = th_0 + (th_1 * inp_data)
	return (hypo)

def cost_func(inp_data, tar_data, th_0, th_1):
	hypo = hypothesis(th_0, th_1, inp_data)
	cost = 1/len(tar_data) * np.power((hypo - tar_data),2)
	#print("calculating cost")
	return cost

def reg_fit(inp_data, tar_data):
	#print("fitting data")
	theta_zero = random.randint(0,10)
	theta_one = random.randint(10,20)
	delta_threshold = 0.00000001
	alpha = 0.00001
	delta = float('inf')
	while (delta >= delta_threshold):
		#print("inside while loop")
		hypo = hypothesis(theta_zero, theta_one, inp_data)
		new_th_0 = theta_zero - alpha * 1/len(inp_data) * sum(hypo - tar_data)
		new_th_1 = theta_one - alpha * 1/len(inp_data) * sum((hypo - tar_data) * inp_data)
		delta = abs(new_th_0 + new_th_1 - theta_zero - theta_one)
		#print(delta)
		theta_zero = new_th_0
		theta_one = new_th_1
	return (theta_zero, theta_one)

def reg_predict(inp_data, th_0, th_1):
	pred = th_0 + th_1 * inp_data
	return pred

def initialize():
	input_file_path = 'C:\\Users\\NISHANTH\\Desktop\\ML_python\\data\\linear_reg.csv'
	input_file_fp = open(input_file_path, 'rb')
	input_data = csv.reader(input_file_fp, delimiter = ',')

	reg_inp_data = []
	reg_tar_data = []

	for row in input_data:
		reg_inp_data.append( row[0] )
		reg_tar_data.append( row[1] )

	reg_inp_data = np.array(reg_inp_data[1:]).astype(float)
	reg_tar_data = np.array(reg_tar_data[1:]).astype(float)

	input_file_fp.close()

	params = reg_fit(reg_inp_data, reg_tar_data)

	print(params)

if(__name__ == '__main__'):
	initialize()
