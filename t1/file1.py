#!/usr/bin/python3

import random

X = [round(random.uniform(0, 100), 2) for i in range(100)]
X = [10. if x < 10. else x for x in X]
X = [90. if x > 90. else x for x in X]
print(f"mean: {round(sum(X)/100, 2)}")
print(f"max: {max(X)}")
print(f"min: {min(X)}")
