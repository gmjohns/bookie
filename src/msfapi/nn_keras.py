import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from keras.models import model_from_json
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt

feature_num = 24

def create_model(init_mode='uniform', node_num=24, input_size=feature_num):
    # define model
    model = Sequential()
    model.add(Dense(node_num, kernel_initializer=init_mode, input_dim=input_size, activation='relu'))
    model.add(Dense(2*node_num, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    # compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def hp_tune_cv(train_file):
    trainset = pd.read_csv(train_file, index_col=0).values
    X_train = trainset[:,0:feature_num]
    y_train = trainset[:,feature_num]

    seed = 7
    np.random.seed(seed)
    batch_size = [256, 512]
    epochs = [10, 25, 50, 100]
    node_num = [8, 24, 64]
    init_mode = ['uniform', 'he_uniform']

    model_CV = KerasClassifier(build_fn=create_model, verbose=1)
    param_grid = dict(epochs=epochs, batch_size=batch_size, init_mode=init_mode, node_num=node_num)
    grid = GridSearchCV(estimator=model_CV, param_grid=param_grid, n_jobs=-1, cv=10)
    grid_result = grid.fit(X_train, y_train)

    # print results
    print(f'Best Accuracy for {grid_result.best_score_} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print(f' mean={mean:.4}, std={stdev:.4} using {param}')


def test_model(train_file, test_file, epochs, batch_size, init_mode, node_num, components):
    trainset = pd.read_csv(train_file, index_col=0).values
    X_train = trainset[:,0:components]
    y_train = trainset[:,feature_num]
    seed = 20
    np.random.seed(seed)
    testset = pd.read_csv(test_file, index_col=0).values
    X_test = testset[:,0:components]
    y_test = testset[:,feature_num]
    model = create_model(init_mode=init_mode, node_num=node_num, input_size=components)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    predictions = model.predict_classes(X_test)
    for i in range(len(predictions)):
        print('%s => %d (expected %d)' % (i, predictions[i], y_test[i]))

    return metrics.accuracy_score(y_test, predictions)


if __name__ == '__main__':
    # hp_tune_cv('data/17-18-standardized.csv')
    epochs=10
    batch_size=512
    init_mode= 'uniform'
    node_num = 24
    std_acc = test_model('data/17-18-standardized.csv', 'data/2019-standardized.csv', epochs=epochs, batch_size=batch_size, init_mode=init_mode, node_num=node_num, components=feature_num)
    norm_acc = test_model('data/17-18-norm-centered.csv', 'data/2019-norm-centered.csv', epochs=epochs, batch_size=batch_size, init_mode=init_mode, node_num=node_num, components=feature_num)
    pca_std_acc = test_model('data/17-18-pca-std.csv', 'data/2019-pca-std.csv', epochs=epochs, batch_size=batch_size, init_mode=init_mode, node_num=node_num, components=5)
    pca_norm_acc = test_model('data/17-18-pca-norm.csv', 'data/2019-pca-norm.csv', epochs=epochs, batch_size=batch_size, init_mode=init_mode, node_num=node_num, components=5)
    
    col_labels = ['PCA', 'No PCA']
    row_labels = ['Standardized', 'Normalized']
    table_vals = [[float(round(pca_std_acc, 4)), float(round(std_acc, 4))], [float(round(pca_norm_acc, 4)), float(round(norm_acc, 4))]]
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