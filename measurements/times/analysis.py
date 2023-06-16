import numpy as np

f = open("times.txt", "rt")

total_time = 0
total_inferences = 0
values = []

for i in range(5):
    val = f.readline() 
while val:
    values.append(float(val))
    total_time = total_time + float(val)
    total_inferences = total_inferences + 1
    val = f.readline()

print("Inferences executed: " + str(total_inferences) + " on a total time of: " + str(total_time) + " s")
print("Execution time statistics: ")
print("Mean: " + str(np.mean(values)))
print("Median: " + str(np.median(values)))
print("Standard Deviation: " + str(np.std(values)))
f.close()

