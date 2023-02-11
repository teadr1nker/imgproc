#!/usr/bin/python3
import numpy as np

temp1 = np.random.uniform(15, 33, 30)
temp2 = np.random.uniform(15, 33, 30)
temp = np.concatenate((temp1[temp1 > 24], temp2[temp2 > 24]))
print(f"len: {len(temp)}")
print(f'min: {temp.min()}')
print(f'max: {temp.max()}')
print(f'mean: {temp.mean()}')
print(f'var: {temp.var()}')
