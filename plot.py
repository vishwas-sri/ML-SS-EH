import matplotlib.pyplot as plt
def show_plot(file): #,mark
    for [fpr,tpr,auc,types] in file:
        plt.plot(fpr, tpr, label='%s = %0.2f' % (types, auc)) #,markevery=mark
        plt.plot([0, 1], [0, 1], 'k--')  # Plot the diagonal line (random classifier)
    plt.xlabel('False Positive Rate (Pf)')
    plt.ylabel('True Positive Rate (Pd)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()