import matplotlib.pyplot as plt
plt.ion()

fig, ax = plt.subplots()

ax.plot([1,2])
fig.show()
fig.save("test")