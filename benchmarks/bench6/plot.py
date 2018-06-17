import matplotlib.pyplot as plt


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


load_and_plot_file("prod opencl with copying data")
load_and_plot_file("prod opencl without copying data")
load_and_plot_file("prod ublas")
plt.xlabel('Size')
plt.ylabel('Time')
plt.title("Time vs size of prod() function")
plt.legend()
plt.show()
