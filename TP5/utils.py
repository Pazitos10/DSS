import string
import numpy as np
from pandas import DataFrame
from sklearn import preprocessing

#Creamos algunas funciones de utilidad
def train_encoder(encoder):
    """
        Crea una lista con las clases presentes en el archivo de entrenamiento:
        Letras: A..S,W,Y
        Luego entrena el encoder con dichas clases y lo retorna
    """
    classes_ = [s for s in string.uppercase[:19]] #Representa todas las letras presentes en el archivo A-S + W + Y
    classes_.append('W')
    classes_.append('Y')
    return encoder.fit(classes_)


def get_alpha(features, dataset, encoder):
    """
        Identificados los features (nombres de las columnas) cuyo valor no es numerico,
        esta funcion aplica una codificacion con el encoder pasado como parametro
        y devuelve una lista con los datos transformados, en sus respectivas columnas.
    """
    alpha_cols = []
    for f in features:
        col = dataset.ix[:, dataset.columns == f]
        alpha_cols.append(np.asarray(col))
    encoded_values = [] 
    for col in alpha_cols:
        result = encoder.transform(col.ravel())
        encoded_values.append(result)
    return encoded_values

def get_non_alpha(features, dataset):
    """
        Identificados los features (nombres de las columnas) cuyo valor es numerico,
        esta funcion separa dichas columnas del dataset original y las devuelve.
    """
    non_alpha_cols = []
    for f in features:
        col = np.asarray(dataset.ix[:, dataset.columns == f])
        non_alpha_cols.append(col.ravel())
    return non_alpha_cols

def get_features_names(values, columns):
    """
        Dados los valores, retorna dos listas:
        * alpha_features: que contendra los titulos de las features con letras
        * non_alpha_features: que contendra los titulos de las features con numeros
    """
    alpha_features = [] 
    non_alpha_features = []
    for i, v in enumerate(values):
        if not str(v).isdigit():
            alpha_features.append(columns[i])
        else:
            non_alpha_features.append(columns[i])
    return alpha_features, non_alpha_features


def preprocess_data(dataset):
    """
        Recibe el dataset original y crea uno nuevo habiendo codificado lo que corresponde,
        es decir, los datos alfanumericos.
    """
    le = preprocessing.LabelEncoder()
    le = train_encoder(le)
    
    a_features, na_features = get_features_names(dataset.values[0], dataset.columns) 

    na_dataset = get_non_alpha(na_features, dataset)
    a_dataset = get_alpha(a_features, dataset, le)
    
    df = {}
    for i, f in enumerate(a_features):
        df.update({f: a_dataset[i]})

    for i, f in enumerate(na_features):
        df.update({f: na_dataset[i]})

    new_dataset = DataFrame(df)
    return new_dataset
