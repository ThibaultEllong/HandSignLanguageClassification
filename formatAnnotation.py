from sklearn import preprocessing
import pandas as pd



    
"""df = pd.read_csv('HandSignLanguageClassification/Dataset/Annotation_gloses.csv', sep='\t')

cols = list(df.columns)
action_class, label = cols[0], cols[1]

names = df[action_class].values
labels = df[label].values"""

def to_categorical(annotation_file):
    data = pd.read_csv(annotation_file, sep="\t")
    cols = list(data.columns)
    data.head()
    action_class, label = cols[0], cols[1]
    data[label] = data[label].apply(lambda x: 0 if x == 'Mono' else 1)
    data.to_csv(annotation_file, index=False)
    

to_categorical('HandSignLanguageClassification/Dataset/Annotation_gloses.csv')