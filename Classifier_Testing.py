import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score,confusion_matrix, recall_score, roc_curve,roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class Classifier_Testing:
    def __init__(self):
        self.df = pd.read_csv ('diabetes.csv')
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        
        self.xpca_test=[]
        self.xpca_train=[]
        self.xlda_test=[]
        self.xpca_train=[]
        self.tpr=[]
        self.fpr=[]        
        
    def analylise_data(self):
        #show first five data entries in csv
        five_data = self.df.tail(5)
        print(five_data)
        #get statistic info on the data 
        print(self.df.describe())
        datas = self.df.iloc[:,-1] 
        print ("classes:",datas.unique())
        #classification type(binary) 1 or 0 meaning diabetic or non diabetic
        #find out the no of rows for each outcome
        print(self.df.groupby('Outcome').size())
        # create two seperate dataframes for each Outcome result
        diabetic=self.df[self.df['Outcome']== 1]
        non_diabetic=self.df[self.df['Outcome']== 0]
        #the features desciptions of each Outcome
        print(diabetic.describe())
        print(non_diabetic.describe())
       
        
    def train_stand(self):
        #divide the datset into a feature set and corresponding labels
        x = self.df.drop('Outcome', 1)
        y = self.df['Outcome']
        #split training set 75% test set 25%
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, 
                                                    test_size=0.25, random_state=0)
        #print all training data and trsting 
        print(len(self.x_train + self.y_train))
        print(len(self.x_test + self.y_test))
        #rescale the data so that all features are between 0 and 1 using
        #Min Max scaler 
        scaler = MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        
        #standarise x for components comparison
        #the first and second components consistute to approxamply 47% second and fourth %13
        #fourth and fith %2 and sixth and seventh %1
        """ for components analysis"""
        #import seaborn as sns
        #x = scaler.fit_transform(x)
        #print(x)
        #pca = PCA(n_components=2)
        #pca_data=pca.fit_transform(x)
        #print('PCA',x)
        #new_df = pd.DataFrame(pca_data, columns=['pca_1', 'pca_2'])
        #pca_y = pd.concat([new_df, y],axis=1)
        #ev = pca.explained_variance_ratio_
        #print('Components',ev)
        #sns.FacetGrid(pca_y, hue="Outcome", palette="Set1", height=6).map(plt.scatter,"pca_1","pca_2").add_legend()
        #plt.show()
        """end of component graph"""
        
    def show_features(self):
        #generate scatter plot to show the features pregnancies and glucose 
        #range before and after feature scaling
        #before 
        plt.figure(figsize=(8,6))
        plt.scatter(self.df['Pregnancies'],self.df['Glucose'], color='blue',
                    label='orginal data', s=2)
        #after
        plt.scatter(self.x_test[:,0],self.x_test[:,1], color ='yellow',
                    label = 'After Feature Scaling', s=2)
        #scatter plot labelets 
        plt.xlabel('Pregnancies feature')
        plt.ylabel('Glucose Feature')
        plt.grid()
        plt.legend(loc='upper right')
      

#-----------------Feature Reudction Analysis-----------------------------------
    #Lda tested components with explaine varience ratio only one componenet 
    def lda(self):
        print('\nLinear Component Analysis')
        lda = LDA(n_components=1)
        self.xlda_train = lda.fit_transform(self.x_train,  self.y_train)
        self.xlda_test = lda.transform(self.x_test)
        #ev = lda.explained_variance_ratio_
        #print(ev)
        
    #pca tested components with explained variance ration seven components 
    #gives best results
    def pca(self):
         print('\nPrinciple Component Analysis')
         pca = PCA(n_components=7)
         #pass the features set to pca 
         self.xpca_train = pca.fit_transform(self.x_train)
         self.xpca_test = pca.transform(self.x_test)
     
#----------------Generating Reults ---------------------------------------        
    #y_ pred is x/ y_score is y
    #Plot roc_auc graph whith each classifer results 
    def roc_auc(self, x, y, classifier):
        plt.plot([0,1], [0,1], 'k--')
        plt.plot(x,y, label ='auc')
        plt.xlabel('False positive rate')
        plt.ylabel('True posituver rate')
        plt.title(classifier)
        plt.grid()
        plt.legend(loc='upper right')
        plt.show()
    #y_ pred is x/ y_score is y
    #generate confusion matrix, accuracy, roc curve score, sensitivity and specificity    
    def get_results(self, x, y):
        cm = confusion_matrix(self.y_test, x)
        print('\nconfusion matrix\n',cm)
        print('Accuracy'+str(accuracy_score(self.y_test, x)))
        self.fpr, self.tpr, threshold = roc_curve(self.y_test, y)
        print('Recall/Sensivity score'+ str(recall_score(self.y_test, x)))
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn+fp)
        print('Specificity',spec)
        print('Score', roc_auc_score(self.y_test, y))
        
ct = Classifier_Testing()
ct.analylise_data()
ct.train_stand()
#ct.show_features()

from sklearn.neighbors import KNeighborsClassifier
class Knn_Testing(Classifier_Testing):
    def __init__(self):
        Classifier_Testing.__init__(self)
        self.y_pred = []
        self.y_score=[]
#------------------------K Nearest Neighbor-----------------------------------
    #able to test k with GUI with user input 
    def k_n_n(self, x, y):
        knn = KNeighborsClassifier(n_neighbors=11)
        knn.fit(x,self.y_train)
        self.y_pred = knn.predict(y)
        self.y_score = knn.predict_proba(y)[:,1]
       # return self.y_pred, self.y_score
     #Method for train knn on it own 
    def knn_own(self):
        print('KNN without analysis')
        self.k_n_n(self.x_train, self.x_test)
        self.get_results(self.y_pred, self.y_score)
        self.roc_auc(self.fpr, self.tpr,'knn(n_neighbors)=11' )
    #KNN with linear discriminent analysis
    def knn_lda(self):
        self.lda()
        self.k_n_n(self.xlda_train, self.xlda_test)
        self.get_results(self.y_pred, self.y_score)
        self.roc_auc(self.fpr, self.tpr,'lda:knn(n_neighbors)=11' )
    #KNN with linear discriminent analysis
    def knn_pca(self):
         self.pca()
         self.k_n_n(self.xpca_train, self.xpca_test)
         self.get_results(self.y_pred,self.y_score)
         self.roc_auc(self.fpr, self.tpr,'pca:knn(n_neighbors)=11')

knn= Knn_Testing()
knn.train_stand()
knn.knn_own()
knn.knn_lda()
knn.knn_pca()
#---------------------Naive Bayes---------------------------------------------
from sklearn.naive_bayes import GaussianNB
class Nb_Testing(Classifier_Testing):
    def __init__(self):
        Classifier_Testing.__init__(self)
        self.yn_pred=[]
        self.yn_scores=[]
        #call train and standisation method
        Classifier_Testing.train_stand(self)

    #Method for train knn on it own
    # x feature train and y x testing          
    def n_b(self, x, y):
         #Classifier_Testing.train_stand(self)
         clf = GaussianNB()
         clf.fit(x, self.y_train)
         self.yn_pred = clf.predict(y)
         self.yn_scores = clf.predict_proba(y)[:,1]
    #train naive bayes on own     
    def naive_bayes(self):
        print('Naive Bayes without ananysis')
        self.n_b(self.x_train, self.x_test)
        self.get_results(self.yn_pred, self.yn_scores)
        self.roc_auc(self.fpr, self.tpr, 'Naive Bayes' )
    #train naive bayes with principle component anaylsis    
    def nb_pca(self):
        self.pca()
        self.n_b(self.xpca_train, self.xpca_test)
        self.get_results(self.yn_pred, self.yn_scores)
        self.roc_auc(self.fpr, self.tpr, 'Naive Bayes:pca')
    #train naive bayes with linear discriminent analysis
    def nb_lda(self):
        self.lda()
        self.n_b(self.xlda_train, self.xlda_test)
        self.get_results(self.yn_pred, self.yn_scores)
        self.roc_auc(self.fpr, self.tpr, 'Naive Bayes:lda')
        
nb= Nb_Testing()
nb.naive_bayes()
nb.nb_lda()
nb.nb_pca()

from sklearn.neural_network import MLPClassifier
#--------------------MultiLayer Perceptron------------------------------------
class Mlp_Testing(Classifier_Testing):
    def __init__(self):
        Classifier_Testing.__init__(self)
        self.ym_pred=[]
        self.ym_scores=[]
        #call train and standisation method
        Classifier_Testing.train_stand(self)
        
    def m_l_p(self, x, y):
        mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000, random_state=1)
        mlp.fit(x, self.y_train)
        self.ym_pred = mlp.predict(y)
        self.ym_scores = mlp.predict_proba(y)[:,1]
    
    def mlp(self):
        print('MULTI LAYER PERCEPTRON without analysis')
        self.m_l_p(self.x_train, self.x_test)
        self.get_results(self.ym_pred, self.ym_scores)
        self.roc_auc(self.fpr, self.tpr, 'Multi Layer Perceptron')
        
    #mlp trained with linear disrim and principle component anaylis           
    def mlp_lda(self):
        self.lda()
        self.m_l_p(self.xlda_train, self.xlda_test)
        self.get_results(self.ym_pred, self.ym_scores)
        self.roc_auc(self.fpr, self.tpr, 'Multi layer perceptron:lda')     
           
    def mlp_pca(self):
        self.pca()
        self.m_l_p(self.xpca_train, self.xpca_test)
        self.get_results(self.ym_pred, self.ym_scores)
        self.roc_auc(self.fpr, self.tpr, 'Multi Layer Perceptron:pca')
        
mlp = Mlp_Testing()
mlp.mlp()
mlp.mlp_lda()
mlp.mlp_pca()
