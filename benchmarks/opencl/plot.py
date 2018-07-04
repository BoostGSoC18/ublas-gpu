import matplotlib.pyplot as plt
import sys

def load_and_plot_file(file_name):
    file = open(file_name, "r")
    x = [0]
    y = [0]
    for line in file:
        arr = line.split()
        size = int(arr[0])
        time = int(arr[1])
        x.append(int(size))
        y.append(int(time))
    plt.plot(x, y, label=file_name)


for file in sys.argv[1:]:
    load_and_plot_file(file)
plt.xlabel('Size')
plt.ylabel('Time')
plt.title("Time vs size of function")
plt.legend()
plt.show()
