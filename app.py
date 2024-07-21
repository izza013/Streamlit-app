import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import streamlit as st
import scipy as sc 
import time
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import seaborn as sns 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

st.title('PREDICTION APP')
st.markdown('This app will provide Data Visualization and Analysis')
image = Image.open('/n.png')
st.image(image, caption='ML', use_column_width=True)

option = st.sidebar.selectbox('Choose', ('EDA', 'Data Visualization', 'Model', 'About Us'))

if option == 'EDA':
    st.subheader('Exploratory Data Analysis')
    upload = st.file_uploader('Choose a file', type='csv')
    if upload is not None:
        data = pd.read_csv(upload)
        st.write(data.head(10))
        st.success('File uploaded')
        if st.checkbox('Display shape'):
            st.write(data.shape)
        if st.checkbox('Display columns'):
            st.write(data.columns)
        if st.checkbox('Select columns'):
            col = st.multiselect('Select columns', data.columns)
            df1 = data[col]
            st.dataframe(df1)
        if st.checkbox('Display summary'):
            st.write(data.describe())
        if st.checkbox('Display null values'):
            st.write(data.isnull().sum())
        if st.checkbox('Display datatypes'):
            st.write(data.dtypes)
        if st.checkbox('Display correlation'):
            st.write(df1.corr())
    else:
        st.error('No file uploaded')

elif option == 'Data Visualization':
    st.subheader('Data Visualization')
    upload = st.file_uploader('Choose a file', type='csv')
    if upload is not None:
        data = pd.read_csv(upload)
        st.write(data.head(10))
        st.success('File uploaded')
        if st.checkbox('Select columns'):
            col = st.multiselect('Select columns', data.columns)
            df1 = data[col]
            st.dataframe(df1)
        if st.checkbox('Heatmap'):
            fig, ax = plt.subplots()
            sns.heatmap(df1.corr(), vmax=1, square=True, annot=True, ax=ax)
            st.pyplot(fig)
        if st.checkbox('Scatter plot'):
            fig, ax = plt.subplots()
            sns.scatterplot(data=df1)
            st.pyplot(fig)
        if st.checkbox('Pairplot'):
            pairplot_fig = sns.pairplot(data, diag_kind='kde')
            st.pyplot(pairplot_fig.fig)
        if st.checkbox('Boxplot'):
            fig, ax = plt.subplots()
            sns.boxplot(data=data, orient='h', ax=ax)
            st.pyplot(fig)
        if st.checkbox('Pie chart'):
            all_cols = data.columns.tolist()
            pie_choice = st.selectbox('Select column', all_cols)
            if pie_choice:
                pie_data = data[pie_choice].value_counts()
                fig, ax = plt.subplots()
                ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
    else:
        st.error('No file uploaded')

elif option == 'Model':
    st.subheader('Model Building')
    upload = st.file_uploader('Choose a file', type='csv')
    if upload is not None:
        data = pd.read_csv(upload)
        st.write(data.head(10))
        st.success('File uploaded')
        if st.checkbox('Select multiple columns'):
            new = st.multiselect('Select multiple columns', data.columns)
            df1 = data[new]
            st.dataframe(df1)
            x = df1.iloc[:,0:-1]
            y = df1.iloc[:,-1]

            
            s2 = st.sidebar.selectbox('Select model', ('SVM', 'KNN', 'LR','Decision Tree'))

            def add_para(algo):
                params = dict()
                if algo == 'SVM':
                    c = st.sidebar.slider('C', 0.01, 15.0)
                    params['C'] = c
                    gamma = st.sidebar.slider('Gamma', 0.01, 1.0)
                    params['gamma'] = gamma
                    degree = st.sidebar.slider('Degree', 1, 10)
                    params['degree'] = degree
                elif algo == 'KNN':
                       K=st.sidebar.slider('k',3,20)
                       params['k']=K
                       p = st.sidebar.slider('P', 1, 5)
                       params['p'] = p
                       leaf_size = st.sidebar.slider('Leaf size', 1, 50)
                       params['leaf_size'] = leaf_size
                 
                elif algo == 'LR':
                   
                    max_iter = st.sidebar.slider('Max iterations', 100, 1000)
                    params['max_iter'] = max_iter
                

                return params

            params = add_para(s2)
            def get_classifier(algo,parmas):
               clf=None
               if algo=='SVM':
                clf=SVC(C=parmas['C'])
                clf=SVC(gamma=params['gamma'])
                clf=SVC(degree=params['degree'])
               elif algo=='KNN':
                 clf=KNeighborsClassifier(n_neighbors=parmas['k'])
                 clf=KNeighborsClassifier(p=parmas['p'])
                 clf=KNeighborsClassifier(leaf_size=params['leaf_size'])
               elif algo=='Decision Tree':
                   DecisionTreeClassifier()
               else:
                   algo=='LR'
                   
                   clf=LogisticRegression(max_iter=params['max_iter'])
               return clf
            clf=get_classifier(s2, params)


            x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20,test_size=0.2)
            clf.fit(x_train,y_train)
            y_pred=clf.predict(x_test)
            st.write(y_pred)

            accuracy=accuracy_score(y_test,y_pred)*100
            st.write('Algo name : ',s2)
            st.write('accuracy is : ',accuracy)

else:
   st.subheader('About Us')
   st.write('This is a web app to predict the accuracy of the model using different algorithms along with EDA and Data Visulaization ')
     
