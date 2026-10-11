import random
import torch
from torch import nn
import matplotlib.pyplot as plt
from pid_template import make_car, update, calculate_desired_acceleration, acceleration_to_throttle_percentage

random.seed(0)
torch.manual_seed(42)

K_P = 1.0
STEPS = 2000

car = make_car(desired_v=20.0, dt=0.1)
X = []
Y = []

for step in range(STEPS):
    if step % 100 == 0:
        car["desired_v"] = random.uniform(0, 30)

    X.append([car["v"], car["desired_v"]])
    accel, error = calculate_desired_acceleration(car, K_P, 0, 0)
    Y.append([accel])

    throttle = acceleration_to_throttle_percentage(accel)
    update(car, throttle)

X = torch.tensor(X, dtype = torch.float32)
Y = torch.tensor(Y, dtype = torch.float32)
print("X shape: ", X.shape, " Y shape: ", Y.shape)
print("first 3 rows:", X[:3].tolist(), Y[:3].tolist())