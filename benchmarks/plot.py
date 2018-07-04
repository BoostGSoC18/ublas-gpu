import matplotlib.pyplot as plt
import sys
import ntpath


def load_and_plot_file(file_path):
    file = open(file_path, "r")
    x = [0]
    y = [0]
    for line in file:
        arr = line.split()
        size = int(arr[0])
        time = int(arr[1])
        x.append(int(size))
        y.append(int(time))
    plt.plot(x, y, label=ntpath.basename(file_path))


for file_path in sys.argv[1:]:
    load_and_plot_file(file_path)
plt.xlabel('Size')
plt.ylabel('Time')
plt.title("Time vs size of function")
plt.legend()
plt.show()
