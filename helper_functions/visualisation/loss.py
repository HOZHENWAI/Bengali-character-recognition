from matplotlib import pyplot as plt

def plot_loss_m(his, epoch, title, dir):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')
    plt.plot(np.arange(0, epoch), his.history['graph_g168_loss'], label='train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['graph_v11_loss'], label='train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['graph_c7_loss'], label='train_consonant_loss')

    plt.plot(np.arange(0, epoch), his.history['val_graph_g168_loss'], label='val_train_root_loss')
    plt.plot(np.arange(0, epoch), his.history['val_graph_v11_loss'], label='val_train_vowel_loss')
    plt.plot(np.arange(0, epoch), his.history['val_graph_c7_loss'], label='val_train_consonant_loss')

    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(dir+'loss_hist_'+str(epoch)+'.png')

def plot_acc_m(his, epoch, title, dir):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['graph_g168_accuracy'], label='train_root_acc')
    plt.plot(np.arange(0, epoch), his.history['graph_v11_accuracy'], label='train_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['graph_c7_accuracy'], label='train_consonant_accuracy')

    plt.plot(np.arange(0, epoch), his.history['val_graph_g168_accuracy'], label='val_root_acc')
    plt.plot(np.arange(0, epoch), his.history['val_graph_v11_accuracy'], label='val_vowel_accuracy')
    plt.plot(np.arange(0, epoch), his.history['val_graph_c7_accuracy'], label='val_consonant_accuracy')

    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.savefig(dir+'acc_hist_'+str(epoch)+'.png')

def plot_loss_o(his, epoch, title, dir):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

    plt.plot(np.arange(0, epoch), his.history['val_loss'], label='val_loss')

    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(dir+'loss_hist_'+str(epoch)+'.png')

def plot_acc_o(his, epoch, title, dir):
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['accuracy'], label='train_acc')

    plt.plot(np.arange(0, epoch), his.history['val_accuracy'], label='val_acc')

    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.savefig(dir+'acc_hist_'+str(epoch)+'.png')
