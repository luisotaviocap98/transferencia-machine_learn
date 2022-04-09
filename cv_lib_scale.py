# http://computacaointeligente.com.br/outros/intro-sklearn-part-3/
import re
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
# from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# https://medium.com/@EduardoSaverin/confusion-matrix-614be4ff4c9e
def saveConfusion(matrix, modelo, y, y_pred, function):
    with open('{}_matriz.txt'.format(modelo),'w') as data:
        function(data, matrix, y, y_pred)
        

def matriz2(data, matrix, y, y_pred):
    data.write("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1: {}".format(metrics.accuracy_score(y, y_pred), 
                                                                                         metrics.precision_score(y, y_pred, average='binary'), metrics.recall_score(y, y_pred, average='binary'),
                                                                                         metrics.f1_score(y, y_pred,average='binary')))
    data.write('\n       _____________Predito_________\n')
    data.write('         _____________| Baixa | Alta |\n')
    data.write('         |Baixa       |'+str(matrix[0][0])+'|'+str(matrix[0][1])+'\n')
    data.write('Original |---------------------------\n')
    data.write('         |Alta        |'+str(matrix[1][0])+'|'+str(matrix[1][1])+'\n')
    
def matriz3(data, matrix, y, y_pred):
    data.write("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1: {}".format(metrics.accuracy_score(y, y_pred), 
                                                                                         metrics.precision_score(y, y_pred, average='weighted'), metrics.recall_score(y, y_pred, average='weighted'),
                                                                                         metrics.f1_score(y, y_pred,average='weighted')))
    data.write('\n       _____________Predito__________________\n')
    data.write('         _____________| Baixa | Média | Alta  |\n')
    data.write('         |Baixa       |'+str(matrix[0][0])+'|'+str(matrix[0][1])+'|'+str(matrix[0][2])+'\n')
    data.write('Original |-------------------------------------\n')
    data.write('         |Média       |'+str(matrix[1][0])+'|'+str(matrix[1][1])+'|'+str(matrix[1][2])+'\n')
    data.write('         --------------------------------------\n')
    data.write('         |Alta        |'+str(matrix[2][0])+'|'+str(matrix[2][1])+'|'+str(matrix[2][2])+'\n')
    
def matriz5(data, matrix, y, y_pred):
    data.write("Accuracy: {} \nPrecision: {} \nRecall: {} \nF1: {}".format(metrics.accuracy_score(y, y_pred), 
                                                                                         metrics.precision_score(y, y_pred, average='weighted'), metrics.recall_score(y, y_pred, average='weighted'),
                                                                                         metrics.f1_score(y, y_pred,average='weighted')))
        
    data.write('\n       ___________________________Predito_____________________________\n')
    data.write('         _____________|Muito Baixa | Baixa | Média | Alta | Muito Alta |\n')
    data.write('         |Muito Baixa |'+str(matrix[0][0])+'|'+str(matrix[0][1])+'|'+str(matrix[0][2])+'|'+str(matrix[0][3])+'|'+str(matrix[0][4])+'\n')
    data.write('         --------------------------------------------------------------\n')
    data.write('         |Baixa       |'+str(matrix[1][0])+'|'+str(matrix[1][1])+'|'+str(matrix[1][2])+'|'+str(matrix[1][3])+'|'+str(matrix[1][4])+'\n')
    data.write('Original |-------------------------------------------------------------\n')
    data.write('         |Média       |'+str(matrix[2][0])+'|'+str(matrix[2][1])+'|'+str(matrix[2][2])+'|'+str(matrix[2][3])+'|'+str(matrix[2][4])+'\n')
    data.write('         --------------------------------------------------------------\n')
    data.write('         |Alta        |'+str(matrix[3][0])+'|'+str(matrix[3][1])+'|'+str(matrix[3][2])+'|'+str(matrix[3][3])+'|'+str(matrix[3][4])+'\n')
    data.write('         --------------------------------------------------------------\n')
    data.write('         |Muito Alta  |'+str(matrix[4][0])+'|'+str(matrix[4][1])+'|'+str(matrix[4][2])+'|'+str(matrix[4][3])+'|'+str(matrix[4][4])+'\n')


def splitter(df):
    features = df.loc[:, df.columns != 'nt_geral_categoria']

    X = features
    y = df['nt_geral_categoria'] #variavel alvo
    
    return X,y

def getCV():
    cv = KFold(n_splits=4, random_state=1, shuffle=True) #25% teste
    return cv

def getData():
    df = pd.read_csv('enade_classifier.csv',sep=',',decimal='.')    
    return df

# def runCV(classificador, X,y,cv, X_train, y_train):    
#     # classificador.fit(X_train, y_train)
#     y_pred = cross_val_predict(classificador, X, y, cv=cv, n_jobs=-1, verbose=15)
    
#     return y_pred


def normalizador(dados):
    anos = list(dados['ano'].unique())
    cursos = list(dados['nome_curso'].unique())
    param_norm = {}
    for i in anos:
        for j in cursos:
            df = dados['nota_geral'].loc[(dados.ano == i) & (dados.nome_curso == j)]
            param_norm[(i,j)]={'min':df.min(),'max':df.max()}
            df = ((df - df.min())/(df.max() - df.min()))
            dados['nota_geral'].loc[(dados.ano == i) & (dados.nome_curso == j)] = df
    return param_norm
            
def normalizarTeste(dados, params):
    anos = list(dados['ano'].unique())
    cursos = list(dados['nome_curso'].unique())
    for i in anos:
        for j in cursos:
            min = params.get((i,j),0).get('min',0)
            max = params.get((i,j),0).get('max',0)
            df = dados['nota_geral'].loc[(dados.ano == i) & (dados.nome_curso == j)]
            df = ((df - min)/(max - min))
            # df = ((df - df.min())/(df.max() - df.min()))
            dados['nota_geral'].loc[(dados.ano == i) & (dados.nome_curso == j)] = df


def getQuantile(df):
    muito_baixa = df['nota_geral'].quantile(.20)
    baixa = df['nota_geral'].quantile(.40)
    media = df['nota_geral'].quantile(.60)
    alta = df['nota_geral'].quantile(.80)
    muito_alta = df['nota_geral'].quantile(.100)
    cinquenta = df['nota_geral'].quantile(.50)
    metade = df['nota_geral'].mean()
    return muito_baixa, baixa, media, alta, muito_alta, cinquenta, metade

def classes2_60(df):
    _, _, media, _,_,_,_ = getQuantile(df)
    bins = [0.0, media, 1.0]
    labels = ['Baixa', 'Alta']
    df['nota_geral'] = pd.cut(df['nota_geral'], bins, labels = labels,include_lowest = True)
    df['nota_geral'].replace({'Baixa':1,'Alta':2},inplace=True) #privada
    return bins
    
def classes2_60_teste(df, bins):
    labels = ['Baixa', 'Alta']
    df['nota_geral'] = pd.cut(df['nota_geral'], bins, labels = labels,include_lowest = True)
    df['nota_geral'].replace({'Baixa':1,'Alta':2},inplace=True) #privada

def classes2(df):
    _, _, _, _,_,cinquenta,metade = getQuantile(df)
    bins = [0.0,(metade+cinquenta)/2, 1.0]
    labels = ['Baixa', 'Alta']
    df['nota_geral'] = pd.cut(df['nota_geral'], bins, labels = labels,include_lowest = True)
    df['nota_geral'].replace({'Baixa':1,'Alta':2},inplace=True) #privada
    return bins
    
def classes2_teste(df,bins):
    labels = ['Baixa', 'Alta']
    df['nota_geral'] = pd.cut(df['nota_geral'], bins, labels = labels,include_lowest = True)
    df['nota_geral'].replace({'Baixa':1,'Alta':2},inplace=True) #privada

def classes3(df):
    bins = [0.0,df['nota_geral'].quantile(.33), df['nota_geral'].quantile(.67), 1.0]
    labels = ['Baixa', 'Média', 'Alta']
    df['nota_geral'] = pd.cut(df['nota_geral'], bins, labels = labels,include_lowest = True)
    df['nota_geral'].replace({'Baixa':1,'Média':2, 'Alta':3},inplace=True) #privada
    return bins
    
def classes3_teste(df,bins):
    # bins = [0.0,df['nota_geral'].quantile(.33), df['nota_geral'].quantile(.67), 1.0]
    labels = ['Baixa', 'Média', 'Alta']
    df['nota_geral'] = pd.cut(df['nota_geral'], bins, labels = labels,include_lowest = True)
    df['nota_geral'].replace({'Baixa':1,'Média':2, 'Alta':3},inplace=True) #privada

def classes5(df):
    muito_baixa, baixa, media, alta,_,_,_ = getQuantile(df)
    bins = [0.0,muito_baixa,baixa, media, alta, 1.0]
    labels = ['Muito Baixa','Baixa', 'Média', 'Alta', 'Muito Alta']
    df['nota_geral'] = pd.cut(df['nota_geral'], bins, labels = labels,include_lowest = True)
    df['nota_geral'].replace({'Muito Baixa':0,'Baixa':1, 'Média':2, 'Alta':3, 'Muito Alta':4},inplace=True) #privada
    return bins

def classes5_teste(df,bins):
    labels = ['Muito Baixa','Baixa', 'Média', 'Alta', 'Muito Alta']
    df['nota_geral'] = pd.cut(df['nota_geral'], bins, labels = labels,include_lowest = True)
    df['nota_geral'].replace({'Muito Baixa':0,'Baixa':1, 'Média':2, 'Alta':3, 'Muito Alta':4},inplace=True) #privada



# https://machinelearningmastery.com/out-of-fold-predictions-in-machine-learning/
# https://www.knowledgehut.com/blog/data-science/k-fold-cross-validation-in-ml
# https://towardsdatascience.com/cross-validation-a-beginners-guide-5b8ca04962cd
def executarModelo(cv, model, dataframe, gerar_classe, classe_teste, params):
    data_y, data_ypred = list(), list()
    # enumerate splits
    X = dataframe.loc[:, dataframe.columns != 'nota_geral']
    y = dataframe['nota_geral']
    
    i=1
    for train_ix, test_ix in cv.split(X,y):
        # get data
        print('Split', i)
        print('\ntreino {} e teste {}'.format(len(train_ix), len(test_ix)))
        i+=1
        df = dataframe.copy()
        
        recorte_treino = df.loc[train_ix,]
        print('normalizando treino')
        param = normalizador(recorte_treino)
        print('gerando classe treino')
        bins = gerar_classe(recorte_treino)
        df.iloc[train_ix] = recorte_treino
        
        
        recorte_teste = df.loc[test_ix,]
        print('normalizando teste')
        normalizarTeste(recorte_teste,param)
        recorte_teste['nota_geral'] = ((recorte_teste['nota_geral'] - recorte_teste['nota_geral'].min())/(recorte_teste['nota_geral'].max() - recorte_teste['nota_geral'].min()))
        print('classes teste')
        classe_teste(recorte_teste,bins)
        df.iloc[test_ix] = recorte_teste
        
        df['nome_curso'] = df['nome_curso'].replace({'BCC':0, 'SI':1, 'EC':2, 'ADS':3, 'RC':4, 'LCC':5, 'GTI':6}) #privada
        # X = df_copy.loc[:, df_copy.columns != 'nota_geral']        
        # y = df_copy['nota_geral']
        X = df.loc[:, df.columns != 'nota_geral']
        y = df['nota_geral']
        
        
        train_X, test_X = X.iloc[train_ix], X.iloc[test_ix]
        train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
        
        # fit model
        print('fit do modelo')
        # print('nan', train_X.isnull().values.any(), train_y.isnull().values.any())
        # model.fit(train_X, train_y)
        # make predictions
        print('predizendo')
        # print('nan', test_X.isnull().values.any())
        # ypred = model.predict(test_X)
        # store
        clf = GridSearchCV(model, params, cv=StratifiedKFold(n_splits=5))
        clf.fit(train_X, train_y)
        
        #problema de classe nao ser predita, estao ficando com tamanhos diferentes
        data_y.extend(pd.Series(test_y, dtype=object).dropna().tolist().copy())
        data_ypred.extend(pd.Series(clf.predict(test_X), dtype=object).dropna().tolist().copy())
        print()
    
    return data_y, data_ypred


def executarTreinoTeste(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,stratify=y.tolist())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25,stratify=y.tolist())


def main():
    pd.options.mode.chained_assignment = None
    divisoes = ['2','2_60','3','5']
    
    configuracoes = {
        '2': {
            'labels': [1,2],
            'function': matriz2,
            'classes': classes2,
            'classes_teste': classes2_teste
        },
        '2_60': {
            'labels': [1,2],
            'function': matriz2,
            'classes': classes2_60,
            'classes_teste': classes2_60_teste
        },
        '3': {
            'labels': [1,2,3],
            'function': matriz3,
            'classes': classes3,
            'classes_teste': classes3_teste
        },
        '5': {
            'labels': [0,1,2,3,4],
            'function': matriz5,
            'classes': classes5,
            'classes_teste': classes5_teste
        }
    }
    
    classificadores = [
        ['Arvore_classifier', DecisionTreeClassifier(), {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']  
        ,'max_features': ['auto', 'sqrt', 'log2']
        ,'min_samples_leaf': [1,2,3]
        ,'max_depth': [ 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]
        ,'max_leaf_nodes': [ 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]
        ,'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
        } ],
        ['Naive_classifier', GaussianNB(), {'var_smoothing' : [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}],
        ['KNN_classifier', KNeighborsClassifier(), {'weights': ['uniform', 'distance']
        ,'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        ,'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        ,'p': [1,2,3]
        ,'leaf_size': [1, 2, 3, 4, 5, 10, 15, 20, 30, 50]}],
        ['Forest_classifier', RandomForestClassifier(), {'criterion': ['gini', 'entropy']
        ,'class_weight': ['balanced', 'balanced_subsample']
        ,'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
        ,'max_features': ['auto', 'sqrt', 'log2']
        ,'min_samples_leaf': [1,2,3]
        ,'bootstrap': [True,False]
        ,'warm_start': [True,False]
        ,'max_depth': [ 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]
        ,'max_leaf_nodes': [ 2, 3, 4, 5, 10, 15, 20, 30, 50, 100]}],
        ['Logistica_classifier', LogisticRegression(), {'solver': ['newton-cg', 'saga', 'lbfgs']
        ,'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        ,'multi_class': ['auto', 'ovr', 'multinomial']
        ,'warm_start': [True, False]
        ,'C': [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]}],
        ['SVM_classifier', SVC(), {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        ,'gamma': ['scale', 'auto']  
        ,'decision_function_shape': ['ovo', 'ovr']
        ,'cache_size': [100,200,300]
        ,'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        ,'probability': [True, False]
        ,'shrinking': [True, False]
        ,'C': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]}]
    ]
    
    print('Lendo dataset\n')
    df = getData()
    cv = getCV()
    print('Inicio das particoes\n')
    for d in divisoes:
        print(d,'classes \n')
        for i in classificadores:
            # X, y = splitter(df)
            nome_modelo = i[0]
            print('Executando', nome_modelo)
            print()
            classificador = i[1]
            
            # y_pred = runCV(classificador, X,y,cv)
            
            y, y_pred = executarModelo(cv, classificador, df, configuracoes[d]['classes'],
                                       configuracoes[d]['classes_teste'], i[2])

            print('Salvando matriz de confusao')
            saveConfusion(confusion_matrix(y, y_pred, labels=configuracoes[d]['labels']), nome_modelo+'_'+d, y, y_pred, configuracoes[d]['function'])
            print('----------------------------------------- \n')

        
if __name__ == '__main__':
    main()
    # df = getData()
    # print(df['nome_curso'].unique())
    
    
    
# https://towardsdatascience.com/comprehensive-guide-to-multiclass-classification-with-sklearn-127cc500f362
# https://stackoverflow.com/questions/45890328/sklearn-metrics-for-multiclass-classification
# https://machinelearningmastery.com/tour-of-evaluation-metrics-for-imbalanced-classification/
# https://stats.stackexchange.com/questions/233275/multilabel-classification-metrics-on-scikit
# https://www.codegrepper.com/code-examples/python/metrics+for+multiclass+classification+sklearn
# geeksforgeeks.org/multiclass-classification-using-scikit-learn/
# https://www.kaggle.com/nkitgupta/evaluation-metrics-for-multi-class-classification