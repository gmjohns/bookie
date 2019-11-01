import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
import matplotlib.pyplot as plt

def test_model(train_file, test_file, feature_num):
    trainset = pd.read_csv(train_file, index_col=0).values
    X_train = trainset[:,0:feature_num]
    y_train = trainset[:,18]

    testset = pd.read_csv(test_file, index_col=0).values
    X_test = testset[:,0:feature_num]
    y_test = testset[:,18]

    model = Sequential()
    model.add(Dense(8, input_dim=feature_num, activation='relu'))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=150, batch_size=10)

    predictions = model.predict_classes(X_test)
    for i in range(len(predictions)):
        print('%s => %d (expected %d)' % (i, predictions[i], y_test[i]))

    return metrics.accuracy_score(y_test, predictions)

if __name__ == '__main__':
    std_acc = test_model('2017-standardized.csv', '2018-standardized.csv', 18)
    norm_acc = test_model('2017-norm-centered.csv', '2018-norm-centered.csv', 18)
    pca_std_acc = test_model('2017-pca-std.csv', '2018-pca-std.csv', 4)
    pca_norm_acc = test_model('2017-pca-norm.csv', '2018-pca-norm.csv', 4)
    
    col_labels = ['PCA', 'No PCA']
    row_labels = ['Standardized', 'Normalized']
    table_vals = [[float(round(std_acc, 4)), float(round(pca_std_acc, 4))], [float(round(norm_acc, 4)), float(round(pca_norm_acc, 4))]]
    the_table = plt.table(cellText=table_vals,
                      colWidths=[0.1] * 3,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(24)
    the_table.scale(4, 4)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.tick_params(axis='y', which='both', right=False, left=False, labelleft=False)
    for pos in ['right','top','bottom','left']:
        plt.gca().spines[pos].set_visible(False)
    plt.savefig('nn_accuracy.png', bbox_inches='tight', pad_inches=0.05)