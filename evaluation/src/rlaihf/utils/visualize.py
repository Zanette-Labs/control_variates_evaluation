import numpy as np
import matplotlib.pyplot as plt

def plot_alpha_loss(x, y, file_path: str): 

    # Plot ls_var_err as a function of d
    plt.plot(x, y,   marker='o', linestyle='-', color='red')

    # Add labels and title
    plt.xlabel('Alpha')
    plt.ylabel('Validation Error')
    plt.title('Validation error as a function of the alpha')

    # Show plot
    plt.grid(True)
    # plt.show()
    plt.savefig(file_path)