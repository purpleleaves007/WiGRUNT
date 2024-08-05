import matplotlib.pyplot as plt

def show_acc_curv(length, global_train_acc, global_test_acc):
    
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc

    test_x = train_x[length-1::length]
    test_y = global_test_acc

    plt.title('oulucasia RESNET18 ACC')

    plt.plot(train_x, train_y, color='green', label='training accuracy')
    plt.plot(test_x, test_y, color='red', label='testing accuracy')

    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('accs')

    plt.show()

