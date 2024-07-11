import glob
import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import metadata_utility
import utils

# indici trial per lo user-study relativi ai video
# classe_1 = quadrante modello di Russel in alto a destra
classe_1, classe_2, classe_3, classe_4 = metadata_utility.create_df()
classe_1 = np.array(classe_1).squeeze()
classe_2 = np.array(classe_2).squeeze()
classe_3 = np.array(classe_3).squeeze()
classe_4 = np.array(classe_4).squeeze()

user_data = glob.glob('features/*')
user_data.sort()

# all_subject = tutti gli utenti del dataset
all_subject = {'subj_file': user_data}
all_subject = pd.DataFrame(all_subject)
all_subject['subject_id'] = all_subject.index
# assegnazione di id univoco ad ogni utente del dataset
target_unique = all_subject['subject_id'].unique()

Results = {}
CM = []

class_select = classe_1

for idx, subject_ID in enumerate(target_unique):

    # il train ha tutti i soggetti tranne quello su cui si testa il modello
    Subject_train = all_subject.loc[all_subject['subject_id'] != subject_ID].drop(['subject_id'], axis=1)
    Subject_test = all_subject.loc[all_subject['subject_id'] == subject_ID].drop(['subject_id'], axis=1)
    name = Subject_test.iloc[0]['subj_file'].split('\\')[1]

    # importazione di una features, una lista di features o tutte le features

    # features_list = utils.features_list(Subject_train)

    features_list = ['plc']

    X_train = utils.process_subject_data_X(Subject_train, features_list, class_select)
    y_train = utils.process_subject_data_y_2(Subject_train, features_list, class_select)
    X_test = utils.process_subject_data_X(Subject_test, features_list, class_select)
    y_test = utils.process_subject_data_y_2(Subject_test, features_list, class_select)
    X_train['y'] = y_train
    X_test['y'] = y_test
    X_train = X_train.dropna()
    X_test = X_test.dropna()
    y_train = X_train['y']
    y_test = X_test['y']
    X_train = X_train.drop(columns=['y'])
    X_test = X_test.drop(columns=['y'])

    print('Grid_search')

    classifiers = [
        ('SVC', SVC(random_state=42, class_weight='balanced', probability=True, kernel='rbf', gamma='auto')),
        ('RidgeClassifier', RidgeClassifier(class_weight='balanced', random_state=42)),
        ('RandomForest', RandomForestClassifier(random_state=42, class_weight='balanced')),
        ('MLP', MLPClassifier(random_state=42, max_iter=1000, early_stopping=True)),
        ('XGB', XGBClassifier(random_state=42)),
    ]

    param_grids = {
        'SVC': {
            'SVC__C': list(np.arange(1, 100, 5)),
        },
        'RidgeClassifier': {
            'RidgeClassifier__alpha': list(np.arange(1, 1000, 1)),
        },
        'RandomForest': {
            'RandomForest__n_estimators': list(np.arange(50, 200, 50)),
            'RandomForest__max_depth': list(np.arange(5, 50, 5)),
            'RandomForest__max_features': ['auto', 'sqrt', 'log2'],
        },
        'MLP': {
            'MLP__hidden_layer_sizes': [(50,), (100,), (150,)],
            'MLP__activation': ['tanh', 'relu'],
            'MLP__solver': ['adam'],
            'MLP__alpha': list(np.logspace(-4, -1, 4)),
            'MLP__learning_rate': ['constant', 'adaptive'],
        },
        'XGB': {
            'XGB__n_estimators': list(np.arange(50, 100, 1)),
            'XGB__max_depth': list(np.arange(50, 100, 1)),
            'XGB__reg_alpha': list(np.arange(1, 100, 1)),
            'XGB__learning_rate': list(np.arange(0.1, 1, 0.1)),
        }
    }

    common_param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
    }

    # Results = {}

    for clf_name, clf in classifiers:
        print(f"Running Grid Search for {clf_name}")

        pipeline = Pipeline([
            ("scaler", Normalizer()),
            (clf_name, clf)
        ])

        param_grid = {**common_param_grid, **param_grids[clf_name]}

        grid_pipeline = RandomizedSearchCV(pipeline, param_grid, verbose=3, return_train_score=True, cv=3,
                                           random_state=42,
                                           refit=True, n_iter=50)

        grid_pipeline.fit(X_train, y_train)

        result = pd.DataFrame(grid_pipeline.cv_results_)
        result = result.sort_values(['rank_test_score'])

        print('score :', grid_pipeline.score(X_test, y_test))
        preds = grid_pipeline.predict(X_test)
        target_names = ['0', '1']
        cm = confusion_matrix(y_test, preds, labels=grid_pipeline.classes_)
        print(cm)
        CR = classification_report(y_test, preds, target_names=target_names, output_dict=True)
        df_classification_report = pd.DataFrame(CR).transpose()
        print(CR)

        Results.update({name: [result, cm, CR, grid_pipeline]})

path_ = 'plc_quadrante_1.pkl'
pkl.dump(Results, open(path_, 'wb'))
print('')
