import matplotlib.pyplot as plt
import sys
import ntpath


def load_and_plot_file(file_path):
    file = open(file_path, "r")
    x = [0]
    y = [0]
    for line in file:
        if line.strip()[0] == '#':
            continue
        arr = line.split("\t")
        size = float(arr[0].strip())
        time = float(arr[1].strip())
        x.append(float(size))
        y.append(float(time))
    plt.plot(x, y, label=ntpath.basename(file_path))


for file_path in sys.argv[1:]:
    load_and_plot_file(file_path)
plt.xlabel('Size')
plt.ylabel('Time')
plt.title("Time vs size of function")
plt.legend()
plt.show()
