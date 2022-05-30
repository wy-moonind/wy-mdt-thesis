from matplotlib import pyplot as plt

plt.figure(0)

Q = [3.2, 21.5, 29.7, 10.8]
T = [-242, -163, -144.5, -227]

plt.arrow(Q[0], T[0], Q[1] - Q[0], T[1] - T[0],
          head_width=5,
          head_length=8,
          fc='blue',
          ec='blue')
plt.arrow(Q[2], T[2], Q[3] - Q[2], T[3] - T[2],
          head_width=5,
          head_length=8,
          fc='red',
          ec='red')
# plt.scatter(p1[0], p1[1])
# plt.scatter(p2[0], p2[1])

plt.xlim(min(Q) - 50, max(Q) + 50)
plt.ylim(min(T) - 50, max(T) + 50)

plt.xlabel('Q(MW)')
plt.ylabel('Temperature($^\circ$C)')

plt.grid(True)
plt.legend(['Cold', 'Hot'])

plt.show()
