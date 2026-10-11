import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 1.0
K_I = 0
K_D = 0
 
STEPS = 550
 
car = make_car(desired_v=20.0, dt=0.1)

velocities = []
errors = []
times = []

for _ in range(STEPS):
    accel_desired, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
    throttle = acceleration_to_throttle_percentage(accel_desired)
    update(car, throttle)

    velocities.append(car["v"])
    errors.append(error)
    times.append(car["t"])

print(f"Final velocity: {velocities[-1]:.3f} m/s (desired {car['desired_v']})")
print(f"Final error: {errors[-1]:.3f} m/s")

fig, (top, bottom) = plt.subplots(2,1,figsize=(9,7), sharex=True)

top.plot(times, velocities, label="velocity")
top.axhline(car["desired_v"], linestyle="--", color="gray", label="target")
top.set_title(f"Velocity (K_P={K_P}, K_I={K_I}, K_D={K_D})")
top.legend()

bottom.plot(times, errors, color="red")
bottom.axhline(0, linestyle="--", color="gray")
bottom.set_title("Error")
bottom.set_xlabel("time (s)")
bottom.set_ylabel("m/s")

plt.tight_layout()
plt.show()