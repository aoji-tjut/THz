import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_score, recall_score, f1_score

flag_scaler = 1
flag_pca = 1
flag_paint = 1
pca_components = 2
axis = []


def LoadData():
    # X = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/血管/X.txt", dtype=np.float)
    # y = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/血管/y.txt", dtype=np.int)

    X = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/心肌/X.txt", dtype=np.float)
    y = np.loadtxt("/Users/aoji/Documents/THz/data/aj-7-2/训练/心肌/y.txt", dtype=np.int)

    return X, y


def Preprocessing(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape)
    print(X_test.shape)

    if flag_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if flag_pca:
        pca = PCA(pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(X_train.shape)
        print(X_test.shape)
        print(np.sum(pca.explained_variance_ratio_[:]))

    left = np.min(X_train[:, 0])
    right = np.max(X_train[:, 0])
    down = np.min(X_train[:, 1])
    up = np.max(X_train[:, 1])
    axis.append(left - 0.5)
    axis.append(right + 0.5)
    axis.append(down - 0.5)
    axis.append(up + 0.5)
    print(axis)

    # 2d
    if (flag_pca) and (pca_components == 2):
        plt.figure("2 Feature")
        plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
        plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])
        plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1])
        plt.show()

    # 3d
    if (flag_pca) and (pca_components) == 3:
        fig = plt.figure("3 Feature")
        ax = Axes3D(fig)
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], c='r')
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], c='g')
        ax.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], X_train[y_train == 2, 2], c='b')
        plt.show()

    return X_train, X_test, y_train, y_test


def Metrics(model, X_test, y_test):
    y_predict = model.predict(X_test)
    acc = model.score(X_test, y_test)
    precision = precision_score(y_test, y_predict, average="weighted")
    recall = recall_score(y_test, y_predict, average="weighted")
    f1 = f1_score(y_test, y_predict, average="weighted")

    return [acc, precision, recall, f1]


def Boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1)
    )
    x_new = np.c_[x0.ravel(), x1.ravel()]
    y_predict = model.predict(x_new)
    zz = y_predict.reshape(x0.shape)
    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])
    plt.contourf(x0, x1, zz, cmap=custom_cmap)


def Paint(ax, model, score, name, X_train, X_test, y_train, y_test):
    plt.sca(ax)
    plt.title("%s %0.2f" % (name, score))
    Boundary(model, axis)
    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='r', s=10)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='g', s=10)
    plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], c='b', s=10)
    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='r', s=10)
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='g', s=10)
    plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], c='b', s=10)
    plt.axis(axis)
    plt.xticks([])
    plt.yticks([])


def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()
    param_grid = [
        {
            "weights": ["uniform"],
            "n_neighbors": np.linspace(1, 10, 10, dtype=np.int)
        },
        {
            "weights": ["distance"],
            "n_neighbors": np.linspace(1, 10, 10, dtype=np.int),
            'p': np.linspace(1, 6, 6, dtype=np.int)
        }
    ]
    grid_search = GridSearchCV(knn, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


def LOG(X_train, X_test, y_train, y_test):
    log = LogisticRegression()
    param_grid = [
        {
            "penalty": ["l2"],
            "C": np.linspace(0.2, 2.0, 10, dtype=np.float),
            "multi_class": ["multinomial"],
            "solver": ["newton-cg", "lbfgs", "sag"],
            "max_iter": [10000]
        }
    ]
    grid_search = GridSearchCV(log, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


def SVM(X_train, X_test, y_train, y_test):
    svc = SVC()
    param_grid = [
        {
            "kernel": ["linear"],
            "C": np.linspace(1, 50, 25, dtype=np.int)
        },
        {
            "kernel": ["rbf"],
            "C": np.linspace(1, 50, 25, dtype=np.int),
            "gamma": np.linspace(0.01, 1.0, 50, dtype=np.float)
        }
    ]
    grid_search = GridSearchCV(svc, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


def DT(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    param_grid = [
        {
            "criterion": ["gini", "entropy"],
            "max_depth": np.linspace(1, 5, 5, dtype=np.int)
        }
    ]
    grid_search = GridSearchCV(dt, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


def RF(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier()
    param_grid = [
        {
            "criterion": ["gini", "entropy"],
            "max_depth": np.linspace(1, 5, 5, dtype=np.int),
            "bootstrap": ["True", "False"]
        }
    ]
    grid_search = GridSearchCV(rf, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


def ET(X_train, X_test, y_train, y_test):
    et = ExtraTreesClassifier()
    param_grid = [
        {
            "criterion": ["gini", "entropy"],
            "max_depth": np.linspace(1, 5, 5, dtype=np.int),
            "bootstrap": ["True", "False"]
        }
    ]
    grid_search = GridSearchCV(et, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


def VOT(X_train, X_test, y_train, y_test):
    vot = VotingClassifier(estimators=[("log", LogisticRegression()),
                                       ("svc", SVC(probability=True)),
                                       ("dt", DecisionTreeClassifier())])
    param_grid = [
        {
            "voting": ["hard", "soft"]
        }
    ]
    grid_search = GridSearchCV(vot, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


def BAG(X_train, X_test, y_train, y_test):
    bag = BaggingClassifier()
    param_grid = [
        {
            "base_estimator": [DecisionTreeClassifier()],
            "n_estimators": np.linspace(200, 1000, 5, dtype=np.int),
            "max_samples": np.linspace(10, 50, 5, dtype=np.int),
            "bootstrap": ["True", "False"]
        }
    ]
    grid_search = GridSearchCV(bag, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


def AB(X_train, X_test, y_train, y_test):
    ab = AdaBoostClassifier()
    param_grid = [
        {
            "base_estimator": [DecisionTreeClassifier()],
            "n_estimators": np.linspace(200, 1000, 5, dtype=np.int),
            "learning_rate": np.linspace(0.2, 1.0, 5, dtype=np.float)
        }
    ]
    grid_search = GridSearchCV(ab, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


def GB(X_train, X_test, y_train, y_test):
    gb = GradientBoostingClassifier()
    param_grid = [
        {
            "n_estimators": np.linspace(50, 200, 4, dtype=np.int),
            "learning_rate": np.linspace(0.2, 1.0, 5, dtype=np.float),
            "max_depth": np.linspace(1, 5, 5, dtype=np.int)
        }
    ]
    grid_search = GridSearchCV(gb, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    score = Metrics(grid_search, X_test, y_test)

    return grid_search, score


if __name__ == '__main__':
    X, y = LoadData()
    X_train, X_test, y_train, y_test = Preprocessing(X, y)

    # 训练
    model = [np.nan] * 10
    score = [np.nan] * 10
    name = ["KNN", "LogisticRegression", "SVM", "DecisionTree", "RandomForest",
            "ExtraTrees", "Voting", "Bagging", "AdaBoost", "GradientBoosting"]
    model[0], score[0] = KNN(X_train, X_test, y_train, y_test)
    model[1], score[1] = LOG(X_train, X_test, y_train, y_test)
    model[2], score[2] = SVM(X_train, X_test, y_train, y_test)
    model[3], score[3] = DT(X_train, X_test, y_train, y_test)
    model[4], score[4] = RF(X_train, X_test, y_train, y_test)
    model[5], score[5] = ET(X_train, X_test, y_train, y_test)
    model[6], score[6] = VOT(X_train, X_test, y_train, y_test)
    model[7], score[7] = BAG(X_train, X_test, y_train, y_test)
    model[8], score[8] = AB(X_train, X_test, y_train, y_test)
    model[9], score[9] = GB(X_train, X_test, y_train, y_test)
    print("accuracy  precision  recall     f1     model")
    for i in range(10):
        print("%6.2f%10.2f%10.2f%9.2f    %s" % (score[i][0], score[i][1], score[i][2], score[i][3], name[i]))

    # 2d画图
    if flag_paint:
        ax = [np.nan] * 10
        plt.figure("Boundary", (13, 6))
        for i in range(10):
            ax[i] = plt.subplot(2, 5, i + 1)

        for i in range(10):
            print("%d/10 painting %s" % (i + 1, name[i]))
            Paint(ax[i], model[i], score[i][3], name[i], X_train, X_test, y_train, y_test)

    plt.show()
