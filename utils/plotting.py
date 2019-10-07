import matplotlib.pyplot as plt

def plot_training_history(history, keyword=None):
    """ The `keyword` argument specifies the keys in `history.history` that should be 
        plotted. ex. keyword='loss' will plot all the values that's under 'loss' and 'val_loss'
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    for k, v in history.history.items():
        if keyword is None or keyword in k:
            ax.plot(v, label=k)
    plt.legend()
    plt.show()

