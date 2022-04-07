import os, random

# temp, learning rate, epochs

temp_min = 0.1
bs = 0

with open('bounces/bounces.txt', 'r') as f:
    for line in f:
        bs = int(line)
        break

print(bs)

bs += 1

temp_max = 4.0
learning_rate_min = 0.0001
learning_rate_max = 0.1
epochs_min = 1
epochs_max = 20

bounces = 0
max_bounces = 20

while bounces < max_bounces:
    temp = "{0:.2}".format(random.uniform(temp_min, temp_max))
    learning_rate = "{0:.4}".format(random.uniform(learning_rate_min, learning_rate_max))
    os.system(f'python3 train.py {temp} {learning_rate} {random.randrange(epochs_min, epochs_max)} {bs}')
    bs += 1
    with open('bounces/bounces.txt', 'w') as f:
        f.write(str(bs))