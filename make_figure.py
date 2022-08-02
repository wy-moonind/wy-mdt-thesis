from matplotlib import pyplot as plt
from validation import validation

if __name__ == '__main__':
    train_loss = [0.345285714,
                  0.314142857,
                  0.289428571,
                  0.303142857,
                  0.291285714]
    val_loss = [0.340285714,
                0.305428571,
                0.286714286,
                0.305,
                0.294
                ]
    val_r2 = [0.888,
              0.911285714,
              0.922142857,
              0.910714286,
              0.918
              ]

    order = [3, 5, 7, 9, 11, 13, 15]
    layer = [1, 2, 3, 4, 5]
    fig, ax1 = plt.subplots()
    fig.set_size_inches(9, 6, forward=True)

    line1 = ax1.plot(layer, train_loss, 'g-', label='Training loss')
    line2 = ax1.plot(layer, val_loss, 'r-', label='Validation loss')

    ax2 = ax1.twinx()
    line3 = ax2.plot(layer, val_r2, 'b-', label='Validation R2')

    line = line1 + line2 + line3
    labs = [l.get_label() for l in line]
    ax1.legend(line, labs, loc=0)

    ax1.set_ylabel('RMSE loss')
    ax2.set_ylabel('R2')
    ax1.set_ylim(0.26, 0.4)
    ax2.set_ylim(0.8, 0.94)

    plt.grid(True)

    ax1.set_xlabel('Layer number')

    plt.show()
    pass
