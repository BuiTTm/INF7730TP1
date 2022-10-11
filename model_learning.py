#Module d'apprentissage automatique sous forme de librairie
#Auteur: Bui Trung Tin Michel
import numpy as np
import pandas as pd
from sklearn import naive_bayes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Machine learning models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
#Evaluate models
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
#Plotting
import matplotlib.pyplot as plt

class ModelLearning():
    def __init__(self, file) -> None:
        """Loading dataset file into pandas

        Args:
            file (str): file location of the dataset file
        """
        self.df = pd.read_csv(file)
        #Séparation des colonnes du dataframe sous forme X et Y
        self.x = self.df.copy()
        #print(self.x.columns)
        self.x.drop(['UserID', 'SpammerBoolean', 'Unnamed: 0'], axis=1, inplace=True)
        self.y = self.df['SpammerBoolean']
        self.ml_models = {}
        self.ml_models_pred = {}

    def normalize_data(self, return_value=False):
        """Function to normalize the dataset using Z-Score method
        """
        print(50*'=')
        print(f'Features:{self.x}')
        print(self.df.dtypes)
        std_scale = StandardScaler()
        self.scaled_x = pd.DataFrame(std_scale.fit_transform(self.x), columns=self.x.columns)
        if return_value:
            return self.scaled_x, self.y
    
    def get_x(self):
        return self.x
        
    def get_y(self):
        return self.y

    def select_cols(self, cols):
        """Choose specific cols into the dataframe

        Args:
            cols (str): desired cols to retain into the Dataframe for scaled_x df
        """
        self.scaled_x = self.scaled_x[cols]

    def to_csv(self, filename):
        """Function to save current dataframe as csv

        Args:
            filename (str): desired file name to save the Dataframe as
        """
        print(f'Saving dataframe to .{filename}')
        self.scaled_x.to_csv(f'.{filename}')
                
    def split_dataset(self):
        """_summary_
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.scaled_x, self.y, test_size=0.2, random_state=66)
        print(50*'=')
        print('Training Features Shape:', self.X_train.shape)
        print('Training Labels Shape:', self.y_train.shape)
        print('Testing Features Shape:', self.X_test.shape)
        print('Testing Labels Shape:', self.y_test.shape)
        print(50*'=')
    
    def decisiontree_init(self):
        """Decision tree model to be initialized 
        """
        self.ml_models['DecisionTree'] = DecisionTreeClassifier(random_state=66)\
            .fit(self.X_train, self.y_train)#.predict(self.X_test)
        self.ml_models_pred['DecisionTree'] = self.ml_models['DecisionTree'].predict(self.X_test)
        print(f'Training of Decision Tree completed')

    
    def randomforest_init(self):
        """Random Forest model to be initialized 
        """
        self.ml_models['RandomForest'] = RandomForestClassifier(
            n_estimators=700, 
            random_state=66, 
            bootstrap= False, 
            criterion= 'entropy').fit(self.X_train, self.y_train)#.predict(self.X_test)
        self.ml_models_pred['RandomForest'] = self.ml_models['RandomForest'].predict(self.X_test)
        print(f'Training of Random Forest completed')


    def bagging_init(self):
        """Bagging model to be initialized 
        """    
        self.ml_models['Bagging'] = BaggingClassifier(
            n_estimators=900, 
            max_features= 15, 
            max_samples=7000, 
            bootstrap= False, 
            random_state=66).fit(self.X_train, self.y_train)#.predict(self.X_test)
        self.ml_models_pred['Bagging'] = self.ml_models['Bagging'].predict(self.X_test)
        print(f'Training of Bag completed')


    def ada_init(self):
        """Ada model to be initialized 
        """  
        self.ml_models['Ada'] = AdaBoostClassifier(
            n_estimators=800, 
            random_state=66).fit(self.X_train, self.y_train)#.predict(self.X_test)
        self.ml_models_pred['Ada'] = self.ml_models['Ada'].predict(self.X_test)
        print(f'Training of Ada completed')


    def naivebayes_init(self):
        """Naives Bayes model to be initialized
        """
        self.ml_models['NaivesBayes'] = GaussianNB().fit(self.X_train, self.y_train)#.predict(self.X_test)
        self.ml_models_pred['NaivesBayes'] = self.ml_models['NaivesBayes'].predict(self.X_test)

        print(f'Training of NaiveBayes completed')
    
    def predict_models(self):
        for name, model in self.ml_models.items():
            self.ml_models_pred[name] = self.ml_models[name].predict(self.X_test)

    def display_accuracy(self):
        """Function to display accuracy of ML models 
        trained and inserted in dictionary
        """
        for name, model_pred in self.ml_models_pred.items():
            model_acc = accuracy_score(self.y_test, model_pred)
            print(f'{name} accuracy:{model_acc}')

    def confusion_matrix(self):
        """Function to display confusion matrix to
        display True pos, True neg, False pos 
        and False pos.
        """
        for name, model_pred in self.ml_models_pred.items():
            c_matrix = confusion_matrix(self.y_test, model_pred)
            tn = c_matrix[0][0]
            fn = c_matrix[1][0]
            fp = c_matrix[0][1]
            tp = c_matrix[1][1]
            tpr= tp/(tp+fn)
            fpr= fp/(fp+tn)
            print(f'{name} TP Rate: {tpr}')
            print(f'{name} FP Rate: {fpr}')
            
    def f_score(self):
        for name, model_pred in self.ml_models_pred.items():
            f_value = f1_score(self.y_test, model_pred, pos_label= 1, average='binary')
            print(f'{name} f-measure: {f_value}')
    
    def roc(self, fig_filename):
        for name, model_pred in self.ml_models.items():
            y_pred_prob = model_pred.predict_proba(self.X_test)
            fpr, tpr, thresh = metrics.roc_curve(self.y_test, y_pred_prob[:,1], pos_label=1)
            print(f'{name} ROC: {metrics.auc(fpr, tpr)}')
            auc = metrics.roc_auc_score(self.y_test, y_pred_prob[:,1])
            plt.plot(fpr, tpr, color='blue', label=name)
        plt.title('ROC de tous les modeles')
        plt.ylabel('Taux de vrai Positifs')
        plt.xlabel('Taux de faux positifs')
        plt.legend(loc='best')
        plt.savefig(f'./Figures/{fig_filename}.png')
        plt.show()
