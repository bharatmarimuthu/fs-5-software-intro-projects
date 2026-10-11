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


split = int(0.8 * len(X))
X_train, Y_train = X[:split], Y[:split]
X_test, Y_test = X[split:], Y[split:]
print("train:", len(X_train), " test:", len(X_test))

model = nn.Linear(in_features=2, out_features=1)
print("starting weights:", model.weight.data.tolist(), "bias:", model.bias.data.tolist())


loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
epochs = 5000
train_losses, test_losses = [], []

for epoch in range(epochs):
    model.train()
    Y_pred = model(X_train)
    loss = loss_fn(Y_pred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_loss = loss_fn(model(X_test), Y_test)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d} | train loss {loss.item():.4f} | test loss {test_loss.item():.4f}")


print("learned weights:", model.weight.data.tolist(), " bias:", model.bias.data.tolist())
print(f"expected weights: [[{-K_P}, {K_P}]] bias: [0.0]")

with torch.inference_mode():
    Y_test_pred = model(X_test)
fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7))
top.plot(train_losses, label="train_loss")
top.plot(test_losses, label="test loss")
top.set_title("Loss over training")
top.set_xlabel("epoch")
top.legend()

bottom.plot(Y_test[:, 0], label="actual (PID)")
bottom.plot(Y_test_pred[:, 0], "--", label="predicted (model)")
bottom.set_title("Test set: desired acceleration")
bottom.set_xlabel("sample")
bottom.legend()

plt.tight_layout()
plt.show()