import numpy as np

print(len([f'{int(x)}' for x in np.linspace(30, 450, 10)]))
print([f'{int(x)}' for x in np.linspace(30, 450, 10)])
print([f'{int(x)}' for x in np.linspace(3, 30, 7)])
print([f'{int(x)}' for x in np.linspace(3, 30, 7)])
print([f'{x}' for x in range(0, 101, 5) / 100])