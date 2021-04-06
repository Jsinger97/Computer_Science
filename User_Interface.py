
from sklearn.model_selection import train_test_split
from tkinter import Tk, Canvas, Listbox, Label, Button, Entry, Checkbutton, IntVar, SUNKEN, GROOVE
from sklearn.preprocessing import MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_auc_score
import pandas as pd
from sklearn.neural_network import MLPClassifier
from PIL import ImageTk,Image


class User_Interface:
    def __init__(self, master):
        
        self.df = pd.read_csv ('diabetes.csv')
        #intialising empty arrays to store data results 
        #testing/training
        self.x_train = []
        self.x_test = []
        self.y_test = []
        self.y_train = []
        
        self.master = master
        self.master.title('Diabetes Classification')
        self.master.config(background = 'light grey')
        self.master.geometry("800x650+30+30")
        self.create_window(master)
        
    def create_window(self, master):
             #===============Data Analysis Danvas==========
        self.canvas = Canvas(self.master, height= 180, width=380, background = 'light grey',borderwidth=1, relief=GROOVE)
        self.canvas.place(x=10, y=20)
        self.listbox = Listbox(self.master, height = 9, width = 15)
        self.listbox.place(x=40, y=40)
        self.title= Label(self.master, text='Data Analysis', background='light grey')
        self.title.place(x=40,y=20)
        self.title.config(font=('System', 10))
        #split data button     
        self.load_button= Button(self.master, text='Split Data')
        self.load_button.place(x=230, y=160)
        self.load_button.config(background= 'light grey', activebackground= 'grey', command= self.preprocessing)
        #Data Ananysis text boxese
        self.class_type = Entry(self.master, width= 10)
        self.class_type.place(x=280, y=40)
        self.class_typel = Label(self.master, text='DataSet Class Labels', background= 'light grey')
        self.class_typel.place(x=160, y=40)
        self.class_typel.config(font=('Times', 9))
        self.class_typel2 = Label(self.master, text='Count of Data Items', background= 'light grey')
        self.class_typel2.place(x=160, y=60)
        self.class_typel2.config(font=('Times', 9))
        self.class_type2 = Entry(self.master, width= 10)
        self.class_type2.place(x=280, y=60)
        self.test_datat= Entry(self.master, width=10)
        self.test_datat.place(x=280, y=80)
        self.test_data= Label(self.master, text='Testing Data Items', background='light grey')
        self.test_data.place(x=160, y=80)
        self.test_data.config(font=('Times', 9))
        self.train_datat= Entry(self.master, width=10)
        self.train_datat.place(x=280, y=100)
        self.train_data= Label(self.master, text='Training Data Items', background='light grey')
        self.train_data.place(x=160, y=100)
        self.train_data.configure(font=('Times', 9))
            #==============KNN Canas=====================
        self.canvas_1 = Canvas(self.master, height= 180, width=380, background = 'light grey',borderwidth=1, relief=GROOVE)
        self.canvas_1.place(x=405, y=20)
        self.title= Label(self.master, text='K Nearest Neighbor', background='light grey')
        self.title.place(x=440,y=20)
        self.title.config(font=('System', 10))
        #knn checkboxes and labels    
        self.train_l = Label(self.master, text= 'Enter K', background='light grey')
        self.train_l.place(x=440, y=40)
        self.train_l.config(font=('Times', 9))
        self.train_l = Entry(self.master, width= 10)
        self.train_l.place(x=500, y=40)
        self.train2 = Label(self.master, text='Train knn without Analysis', background='light grey')
        self.train2.place(x=440, y=60)
        self.train2.config(font=('Times', 9))
        self.var = IntVar()
        self.train2 = Checkbutton(self.master, background='light grey', variable = self.var)
        self.train2.place(x=660, y=60)
        self.train_l2 = Label(self.master, text='Train with Principal Component Analysis', background='light grey')
        self.train_l2.place(x=440, y=80)
        self.train_l2.config(font=('Times', 9))
        self.var1 = IntVar()
        self.train_l2 = Checkbutton(self.master, background='light grey', variable = self.var1)
        self.train_l2.place(x=660, y=80)
        self.train_l3 = Label(self.master, text='Train with Linear Discriminant Analysis', background='light grey')
        self.train_l3.place(x=440, y=100)
        self.train_l3.config(font=('Times', 9))
        self.var2 = IntVar()
        self.train_l3 = Checkbutton(self.master, background='light grey', variable = self.var2)
        self.train_l3.place(x=660, y=100)
        #knn button
        self.kb= Button(self.master, text='Train')
        self.kb.place(x=560, y=160)
        self.kb.config(background= 'light grey', activebackground= 'grey', command= self.knn_checked)
            #==============Naive Bayes===================
        self.canvas_2 = Canvas(self.master, height= 180, width=380, background = 'light grey',borderwidth=1, relief=GROOVE)
        self.canvas_2.place(x=10, y=220)
        self.ntitle= Label(self.master, text='Naive Bayes', background='light grey')
        self.ntitle.place(x=40,y=220)
        self.ntitle.config(font=('System', 10))
        #naive bayes checkboxes and labels
        self.ntrain_l = Label(self.master, text='Train without Analysis', background='light grey')
        self.ntrain_l.place(x=40, y=260)
        self.ntrain_l.config(font=('Times', 9))
        self.var3 = IntVar()
        self.ntrain_l = Checkbutton(self.master, background='light grey', variable = self.var3)
        self.ntrain_l.place(x=260, y=260)
        self.ntrain_l2 = Label(self.master, text='Train with Principal Component Analysis', background='light grey')
        self.ntrain_l2.place(x=40, y=280)
        self.ntrain_l2.config(font=('Times', 9))
        self.var4 = IntVar()
        self.ntrain_l2 = Checkbutton(self.master, background='light grey', variable = self.var4)
        self.ntrain_l2.place(x=260, y=280)
        self.ntrain_l3 = Label(self.master, text='Train with Linear Discriminant Analysis', background='light grey')
        self.ntrain_l3.place(x=40, y=300)
        self.ntrain_l3.config(font=('Times', 9))
        self.var5 = IntVar()
        self.ntrain_l3 = Checkbutton(self.master, background='light grey', variable = self.var5)
        self.ntrain_l3.place(x=260, y=300)
        #Niave Bayes train button
        self.nb= Button(self.master, text='Train')
        self.nb.place(x=160, y=360)
        self.nb.config(background= 'light grey', activebackground= 'grey', command= self.nb_checked)
            #=============Neaural Network Canvas============
        self.canvas_3 = Canvas(self.master, height= 180, width=380, background = 'light grey',borderwidth=1, relief=GROOVE)
        self.canvas_3.place(x=405, y=220)
        self.nntitle= Label(self.master, text='Neural Network', background='light grey')
        self.nntitle.place(x=440,y=220)
        self.nntitle.config(font=('System', 10))
        #neaural network checkboxes and labels
        self.nntrain_l = Label(self.master, text='Train without Analysis', background='light grey')
        self.nntrain_l.place(x=440, y=260)
        self.nntrain_l.config(font=('Times', 9))
        self.var6 = IntVar()
        self.nntrain_l = Checkbutton(self.master, background='light grey', variable= self.var6)
        self.nntrain_l.place(x=660, y=260)
        self.nntrain_l2 = Label(self.master, text='Train with Principal Component Analysis', background='light grey')
        self.nntrain_l2.place(x=440, y=280)
        self.nntrain_l2.config(font=('Times', 9))
        self.var7 = IntVar()
        self.nntrain_l2 = Checkbutton(self.master, background='light grey', variable=self.var7)
        self.nntrain_l2.place(x=660, y=280)
        self.nntrain_l3 = Label(self.master, text='Train with Linear Discriminant Analysis', background='light grey')
        self.nntrain_l3.place(x=440, y=300)
        self.nntrain_l3.config(font=('Times', 9))
        self.var8 = IntVar()
        self.nntrain_l3 = Checkbutton(self.master, background='light grey', variable=self.var8)
        self.nntrain_l3.place(x=660, y=300)
        #neaural network button 
        self.nn= Button(self.master, text='Train', command = self.mlp_checked)
        self.nn.place(x=570, y=360)
        self.nn.config(background= 'light grey', activebackground= 'grey', )
            #=============Classification Results Canvas======================
        self.canvas_4 = Canvas(self.master, height= 230, width=780, background = 'gray68',borderwidth=1, relief=SUNKEN)
        self.canvas_4.place(x=10, y=410)
        self.rtitle= Label(self.master, text='Classification Report', background='gray68')
        self.rtitle.place(x=20, y=420)
        self.rtitle.config(font=('System', 10))
        #classification results text boxes
        self.class_accl= Label(self.master, text='Accuracy', background='gray68')
        self.class_accl.place(x=20, y=470)
        self.class_accl.configure(font=('Times', 9))
        self.class_acc = Entry(self.master, width= 10)
        self.class_acc.place(x=180, y=470)
        self.class_sens= Label(self.master, text= 'Sensitivity', background='gray68')
        self.class_sens.configure(font=('Times', 9))
        self.class_sens.place(x=20, y=500)
        self.class_sense= Entry(self.master, width=10)
        self.class_sense.place(x=180, y=500)
        self.class_spec= Label(self.master, text= 'Specificity', background='gray68')
        self.class_spec.configure(font=('Times', 9))
        self.class_spec.place(x=20, y=530)
        self.class_spece= Entry(self.master, width=10)
        self.class_spece.place(x=180, y=530)
        self.rocl = Label(self.master, text= 'ROC/AUC Score', background='gray68')
        self.rocl.configure(font=('Times', 9))
        self.rocl.place(x=20, y=560)
        self.rocscore = Entry(self.master, width=10)
        self.rocscore.place(x=180, y=560)
        #auc-roc curve canvas
        self.canvas_im = Canvas(self.master, height= 210, width=260, background = 'gray68',borderwidth=1, relief=SUNKEN)
        self.canvas_im.pack()
        self.canvas_im.place(x=250, y=420) 
        self.canvas_iml= Label(self.master, text='AUC-ROC Curve', background='white',borderwidth=1, relief=SUNKEN)
        self.canvas_iml.place(x=340, y=612)
        self.canvas_iml.configure(font=('Times', 8))
        #scatter graph canvas
        self.canvas_im2 = Canvas(self.master, height= 210, width=260, background = 'gray68',borderwidth=1, relief=SUNKEN)
        self.canvas_im2.pack()
        self.canvas_im2.place(x=520, y=420)
        self.canvas_im2l= Label(self.master, text='X=PCA=1 Y=PCA=2', background='white',borderwidth=1, relief=SUNKEN)
        self.canvas_im2l.place(x=600, y=612)
        self.canvas_im2l.configure(font=('Times', 9))

    #listbox and textboxes for data analysis when split data is clicked   
    def clear_dataset(self):
        self.listbox.delete(0,"end")
        self.class_type.delete(0,"end")
        self.class_type2.delete(0,"end")
    #load diabetes dataset into listbox and data feilds when split data has been clicked  
    def open_file(self):
        self.clear_dataset()
        data_class = self.df.iloc[:,-1]
        d_class = data_class.unique()
        d_count = data_class.count()
        #insert target feature into entry box
        self.class_type.insert("end", d_class)
        #insert count of data items into entry box
        self.class_type2.insert("end", d_count)
        data_list = list(self.df)
        #for everyitem/column in datalist
        for items in data_list:
            #insert colomns into listbox
            self.listbox.insert("end", items)
            
    #preparing the data before classification
    def preprocessing(self):
        self.clear_results()
        #outcome colomn with all rows
        x = self.df.drop('Outcome', 1)
        #just outcome column
        y = self.df['Outcome']
        #split training set 75% test set 25%
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        train_set = len(self.x_train + self.y_train)
        test_set = len(self.x_test + self.y_test)
        #add testing and traing data into text boxs
        self.test_datat.insert("end", test_set)
        self.train_datat.insert("end", train_set)
        scaler = MinMaxScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        #add two components scatter graph
        self.img2=img2 =ImageTk.PhotoImage(Image.open("pca_scatter.png"))
        self.canvas_im2.create_image(0,0,anchor = "nw", image=img2)
#--------------CheckBoxes-------------------------------------------------
    #show all classifier roc-auc results in canvas   
    def knn_checked(self):
        #get if checked
        var = self.var.get()
        var1 = self.var1.get()
        var2 = self.var2.get()
        self.clear_checked()
        self.canvas_im.delete("all")
        #if knn without analysis is checked then show knn roc curve
        if var ==1:
            self.knn_own()
            self.img=img =ImageTk.PhotoImage(Image.open("knn_img.png"))
            self.canvas_im.create_image(0,0,anchor = "nw", image=img)
        #if pca knn is checked then show knn with pca roc curve    
        if var1 == 1:
            self.knn_pca()
            self.img=img =ImageTk.PhotoImage(Image.open("pk_img.png"))
            self.canvas_im.create_image(0,0,anchor = "nw", image=img)
        #if lda knn is checked then show knn with lda roc curve
        if var2 == 1:
            self.knn_lda()
            self.img=img =ImageTk.PhotoImage(Image.open("ldak_img.png"))
            self.canvas_im.create_image(0,0,anchor = "nw", image=img) 
            
    def nb_checked(self):
        var3 = self.var3.get()
        var4 = self.var4.get()
        var5 = self.var5.get()
        self.clear_checked()
        self.canvas_im.delete("all")
        #if naive bayes without analysis is checked then show naive bayes roc curve
        if var3 == 1:
            self.naive_bayes()
            self.img=img =ImageTk.PhotoImage(Image.open("nb_img.png"))
            self.canvas_im.create_image(0,0,anchor = "nw", image=img)
        #if naive bayes with pca is cheked then show naive bayes with pca roc curve
        if var4 == 1:
            self.naiveb_pca()
            self.img=img =ImageTk.PhotoImage(Image.open("pcanb_img.png"))
            self.canvas_im.create_image(0,0,anchor = "nw", image=img)
        #if naive bayes with lda is checked then show naive bayes with lda roc curve
        if var5 == 1:
            self.naiveb_lda()
            self.img=img =ImageTk.PhotoImage(Image.open("ldanb_img.png"))
            self.canvas_im.create_image(0,0,anchor = "nw", image=img)
            
    def mlp_checked(self):
        var6 = self.var6.get()
        var7 = self.var7.get()
        var8 = self.var8.get()
        self.clear_checked()
        self.canvas_im.delete("all")
        
        if var6 ==1:
            self.mlp()
            #''change image to knn without pca
            self.img=img =ImageTk.PhotoImage(Image.open("mlp_img.png"))
            self.canvas_im.create_image(0,0,anchor = "nw", image=img) 
        if var7 == 1:
            self.mlp_pca()
            self.img=img =ImageTk.PhotoImage(Image.open("pcamlp_img.png"))
            self.canvas_im.create_image(0,0,anchor = "nw", image=img)    
        if var8 == 1:
            self.mlp_lda()
            self.img=img =ImageTk.PhotoImage(Image.open("ldamlp_img.png"))
            self.canvas_im.create_image(0,0,anchor = "nw", image=img)          
    #clear all checked button once clicked        
    def clear_checked(self):
        self.var.set(0)
        self.var1.set(0)
        self.var2.set(0)
        self.var3.set(0)
        self.var4.set(0)
        self.var5.set(0)
        self.var6.set(0)
        self.var7.set(0)
        self.var8.set(0)
#------------------end of checkboxes-----------------------------------------
        
    #linear discriminent analysis with one component on training data    
    def lda(self):
        lda = LDA(n_components=1)
        self.xlda_train = lda.fit_transform(self.x_train,  self.y_train)
        self.xlda_test = lda.transform(self.x_test)
        return self.xlda_test, self.xlda_train

    #principle component anaysis with seven components tested to give best results
    def pca(self):
         pca = PCA(n_components=7)
         self.xpca_train = pca.fit_transform(self.x_train)
         self.xpca_test = pca.transform(self.x_test)
         return self.xpca_test, self.xpca_train
        
    #retrive all results for classification report     
    def get_results(self, x, y):
         self.acc =accuracy_score(self.y_test, x)
         #seperate confusion matrix to get true negative, flase positive, false negative and true 
         #true positive for specificity result
         tn, fp, fn, tp = confusion_matrix(self.y_test, x).ravel()
         self.spec = tn / (tn+fp)
         self.roc_score =roc_auc_score(self.y_test, y)
         self.sens = recall_score(self.y_test, x)
#===========================Classifiers=============================================
   #three different classifiers for each model one train without ananysis then pca and lda
    #call methods for training/ splitting and standardisation of data 
    #fit all classifiers with training data 
    #clear all classification results once a classifier has been trained 
     # insert the new classifier resutls into classification report text boxes 
    #-------------------------KNN--------------------------------------------       
    def k_n_n(self, x, y):
        #get input value
         r = int(self.train_l.get())
         #classify knn with input valaue for n_neighbors testing 
         #differnt k values correct k = 11 
         knn = KNeighborsClassifier(n_neighbors=r)
         knn.fit(x,self.y_train)
         self.y_pred = knn.predict(y)
         self.y_score = knn.predict_proba(y)[:,1]
         return self.y_pred, self.y_score
        
    #knn withou anaylisis  
    def knn_own(self):
        self.clear_results()
        self.preprocessing()
        self.k_n_n(self.x_train, self.x_test)
        self.get_results(self.y_pred, self.y_score)
        self.get_results(self.y_pred, self.y_score)
        self.insert_results()
    #knn with linear discriminent analysis    
    def knn_lda(self):
        self.clear_results()
        self.preprocessing()
        self.lda()
        self.k_n_n(self.xlda_train, self.xlda_test)
        self.get_results(self.y_pred, self.y_score)
        self.insert_results()
        
    #knn with principle component analysis                           
    def knn_pca(self):
        self.clear_results()
        self.preprocessing()
        self.pca()
        self.k_n_n(self.xpca_train, self.xpca_test)
        self.get_results(self.y_pred, self.y_score)
        self.insert_results()
#-------------------------Naive Bayes----------------------------------------    
    #train naive bayes
    def n_b(self, x, y):
        clf = GaussianNB()
        clf.fit(x, self.y_train)
        self.yn_pred = clf.predict(y)
        self.yn_scores = clf.predict_proba(y)[:,1]
        return self.yn_pred, self.yn_scores
    #naive bayes without anaylis 
    def naive_bayes(self):
        self.clear_results()
        self.preprocessing()
        self.n_b(self.x_train, self.x_test)
        self.get_results(self.yn_pred, self.yn_scores)
        self.insert_results()
     
    #naive bayes with linear discriminent anaylsis  
    def naiveb_lda(self):
        self.clear_results()
        self.preprocessing()
        self.lda()
        self.n_b(self.xlda_train, self.xlda_test)
        self.get_results(self.yn_pred, self.yn_scores)
        self.insert_results()
    #naive bayes with principle component anaylsis    
    def naiveb_pca(self):
        self.clear_results()
        self.preprocessing()
        self.pca()
        self.n_b(self.xpca_train, self.xpca_test)
        self.get_results(self.yn_pred, self.yn_scores)
        self.insert_results()
#---------------------MultiLayer Perceptron------------------------------------        
         
    #MultiLayer Perceptron method with 100 iterations and ten hidden layers for three nodes    
    def m_l_p(self, x, y):
        mlp = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000, random_state=1)
        mlp.fit(x, self.y_train)
        self.ym_pred = mlp.predict(y)
        self.ym_scores = mlp.predict_proba(y)[:,1]
        return self.ym_pred, self.ym_scores
    
    #empty contents train multilayer perceptron with Principle component Analysis
    #intilize methods
    def mlp(self):
        self.clear_results()
        self.preprocessing()
        self.m_l_p(self.x_train, self.x_test)
        self.get_results(self.ym_pred, self.ym_scores)
        self.insert_results()
        
    def mlp_lda(self):
        self.clear_results()
        self.preprocessing()
        self.lda()
        self.m_l_p(self.xlda_train, self.xlda_test)
        self.get_results(self.ym_pred, self.ym_scores)
        self.insert_results()
       
    def mlp_pca(self):
        self.clear_results()
        self.preprocessing()
        self.pca()
        self.m_l_p(self.xpca_train, self.xpca_test)
        self.get_results(self.ym_pred, self.ym_scores)
        self.insert_results()
    
    def insert_results(self):
        #insert all results into text boxes in class report
        #round results to two decimal places
        self.class_acc.insert("end",self.acc.round(2))
        self.class_sense.insert("end", self.sens.round(2))
        self.class_spece.insert("end", self.spec.round(2))
        self.rocscore.insert("end", self.roc_score.round(2))   
    #clear all results in classification report once another
    #classifier has been selected      
    def clear_results(self):
        self.test_datat.delete(0,"end")
        self.class_acc.delete(0,"end")
        self.train_datat.delete(0,"end")
        self.class_sense.delete(0, "end")
        self.class_spece.delete(0, "end")
        self.rocscore.delete(0,"end")
                
root = Tk()
class_gui = User_Interface(root)
#intialise open file method so listbox and entry boxes are inserted
class_gui.open_file()
root.mainloop() 
        