import matplotlib.pyplot as plt
import sys

if __name__ == "__main__":

    print_path = sys.path[0] + "/print.txt"
    rmse_list = []
    with open(print_path, "r") as print_file:
        while True:
            line = print_file.readline()
            if not line:
                break
            term = line.split(" ")
            if "rmse:" in term:
                rmse = term[term.index("rmse:") + 1]
                rmse_list.append(rmse)
            else:
                continue

    plt.plot(rmse_list[1:])
    plt.show()
