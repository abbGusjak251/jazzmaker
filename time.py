import time
start_time = time.time()
##Din kod
b = 0
for i in range(100000):
    b += 1
print("--- %s seconds ---" % (time.time() - start_time))