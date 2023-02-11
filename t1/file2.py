#!/usr/bin/python3

m = int(input('m: '))
n = int(input('n: '))

mtx = []
for x in range(m):
    row = []
    for y in range(n):
        row.append(min(
            min(abs(y), abs(n-1-y)),
            min(abs(x), abs(m-1-x))
            )+1)
    mtx.append(row)

for row in mtx:
    print(row)
