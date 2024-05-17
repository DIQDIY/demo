# -*- coding: utf-8 -*-
"""
Created on Tue May  7 19:51:40 2024

@author: duanqi
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_curve # 召回曲线
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier  # ada分类器 、 极度随机数分类算法
from sklearn.ensemble import GradientBoostingClassifier # 集成学习梯度提升决策树
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 线性判别分析
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#数据读取
data = pd.read_csv(r"...\WA_Fn-UseC_-Telco-Customer-Churn.csv")
#数据信息显示
print(data.info())
 #   Column            Non-Null Count  Dtype  
#---  ------            --------------  -----  
# 0   customerID        7043 non-null   object 
# 1   gender            7043 non-null   object gender：性别
# 2   SeniorCitizen     7043 non-null   int64  SeniorCitizen：是否是老人
# 3   Partner           7043 non-null   object Partner：是否有伴侣
# 4   Dependents        7043 non-null   object Dependents：是否有需要抚养的孩子
# 5   tenure            7043 non-null   int64  tenure：任职
# 6   PhoneService      7043 non-null   object PhoneService：是否办理电话服务
# 7   MultipleLines     7043 non-null   object MultipleLines：是否开通了多条线路
# 8   InternetService   7043 non-null   object InternetService：是否开通网络服务和开通的服务类型（光纤、电话拨号）
# 9   OnlineSecurity    7043 non-null   object TechSupport：是否办理技术支持服务
# 10  OnlineBackup      7043 non-null   object OnlineBackup：是否办理在线备份服务
# 11  DeviceProtection  7043 non-null   object OnlineSecurity：是否办理在线安全服务
# 12  TechSupport       7043 non-null   object DeviceProtection：是否办理设备保护服务
# 13  StreamingTV       7043 non-null   object StreamingTV：是否办理电视服务
# 14  StreamingMovies   7043 non-null   object StreamingMovies：是否办理电影服务
# 15  Contract          7043 non-null   object Contract：签订合约的时长
# 16  PaperlessBilling  7043 non-null   object PaperlessBilling：是否申请了无纸化账单
# 17  PaymentMethod     7043 non-null   object PaymentMethod：付款方式（电子支票、邮寄支票、银行自动转账、信用卡自动转账）
# 18  MonthlyCharges    7043 non-null   float64 MonthlyCharges：月消费
# 19  TotalCharges      7043 non-null   object TotalCharges：总消费
# 20  Churn             7043 non-null   object Churn：用户是否流失


# 离散字段
category_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                'PaymentMethod']

# 连续字段
numeric_cols = ['MonthlyCharges', 'TotalCharges','tenure']

# 标签
target = 'Churn'

# ID列
ID_col = 'customerID'

#数据预处理
 # 判断ID列是否有重复
data['customerID'].nunique() == data.shape[0]


#TotalCharges存在空值
# 把 TotalCharges 转成数值型 (str类型不能用 astype 转成 float)

data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
print(data.info())
#查看空值数量
print("改变前=",data['TotalCharges'].isnull().sum())
#11个空值
#将空值置为0
data['TotalCharges'].fillna(0, inplace=True)

#查看空值数量
print("改变后=",data['TotalCharges'].isnull().sum())


#判断列内值的分布情况
col_number = ['customerID','tenure','MonthlyCharges','TotalCharges']
for col in data.columns.values:
    if col not in col_number:
        print('列名: {}\n{}\n{}\n'.format(col, '-'*20, data[col].value_counts()))




#对连续型变量进行异常值检验
print(data[numeric_cols].describe())
#数据可视化处理 
        
        

#流失客户样本占比26.5%，留存客户样本占比73.5%，


data_temp = data.iloc[:,1:].copy()
data_dummies = pd.get_dummies(data_temp)
# 标签列转换
data_temp['Churn'] = data_temp['Churn'].apply(lambda x: 1 if x=='Yes' else 0)
data_temp['Churn'].replace(to_replace = 'No', value=0, inplace=True)
# 哑变量转换，将分类变量转换为哑变量，否则无法进行相关系数计算
data_dummies = pd.get_dummies(data_temp)
# 计算相关系数矩阵
print(data_dummies.corr()['Churn'].sort_values(ascending=False))
#相关系数矩阵可视化
sns.set()
plt.figure(figsize=(15,8), dpi=200)
data_dummies.corr()['Churn'].sort_values(ascending=False).plot(kind='bar')
#表明Contract_Month-to-month与Churn之间相关系数为 0.405103，呈现较强的正相关关系
#tenure与Churn之间相关系数-0.352229，呈现较强的负相关关系


#查看是否有被抚养人，是否有伴侣，是否使用无纸化账单	，订购家庭电话服务	绘图
binary_cols=['Dependents','Partner','PaperlessBilling','PhoneService']
fig=plt.figure(figsize=(18,4))
plt.subplots_adjust(hspace=0.3,wspace=0.3)
for i,iterm in enumerate(binary_cols):
    i=i+1
    x=data['Churn'].unique()
    y1=data.Churn[data[iterm]=='Yes'].value_counts()/len(data)
    y2=data.Churn[data[iterm]=='No'].value_counts()/len(data)
    df_i=pd.DataFrame({iterm+'_yes':y1,iterm+'_no':y2})
    axi=fig.add_subplot(1,4,i)
    df_i.plot(kind='bar',ax=axi,stacked=True)
    plt.xticks([0,1],['Retain','Churn'],rotation=1)
    plt.ylabel(iterm)
    plt.title(iterm)
plt.show()

#查看性别绘图
fig=plt.figure(figsize=(10,4))
y1=data.Churn[data.gender=='Female'].value_counts()/len(data)
y2=data.Churn[data.gender=='Male'].value_counts()/len(data)
df_gender=pd.DataFrame({'Female':y1,'Male':y2})
ax1=fig.add_subplot(1,2,1)
df_gender.plot(kind='bar',ax=ax1,stacked=True)
plt.xticks([0,1],['Retain','Churn'],rotation=1)
plt.title('gender')

plt.show()

#查看是否是老人
fig=plt.figure(figsize=(10,4))
y1=data.Churn[data.SeniorCitizen==0].value_counts()/len(data)
y2=data.Churn[data.SeniorCitizen==1].value_counts()/len(data)
df_senior=pd.DataFrame({'Senior_Yes':y1,'Senior_no':y2})
ax2=fig.add_subplot(1,2,2)
df_senior.plot(kind='bar',ax=ax2,stacked=True)
plt.xticks([0,1],['Retain','Churn'],rotation=1)
plt.title('SeniorCitizen')

plt.show()




#查看当前合约类型绘图
x=data.Churn.unique()
y1=data.Churn[data.Contract=='Month-to-month'].value_counts()/len(data)
y2=data.Churn[data.Contract=='One year'].value_counts()/len(data)
y3=data.Churn[data.Contract=='Two year'].value_counts()/len(data)
df_contract=pd.DataFrame({'Month-to-month':y1,'one year':y2,'two year':y3})
df_contract.plot(kind='bar',stacked=True)
plt.xticks([0,1],['Retain','Churn'],rotation=1)
plt.title('Contract')

plt.show()

#查看用户付款方式绘图	
x=data.Churn.unique()
y0=data.Churn[data.PaymentMethod=='Electronic check'].value_counts()/len(data)
y1=data.Churn[data.PaymentMethod=='Mailed check'].value_counts()/len(data)
y2=data.Churn[data.PaymentMethod=='Bank transfer (automatic)'].value_counts()/len(data)
y3=data.Churn[data.PaymentMethod=='Credit card (automatic)'].value_counts()/len(data)
df_paymethod=pd.DataFrame({'Electronic check':y0,'Mailed check':y1,'Bank transfer (automatic)':y2,'Credit card (automatic)':y3})          
df_paymethod.plot(kind='bar',stacked=True)
plt.xticks([0,1],['Retain','Churn'],rotation=1)
plt.title('PaymentMethod')

plt.show()

#查看订购网络服务绘图
x=data.Churn.unique()
y1=data.Churn[data.InternetService=='No'].value_counts()/len(data)
y2=data.Churn[data.InternetService=='DSL'].value_counts()/len(data)
y3=data.Churn[data.InternetService=='Fiber optic'].value_counts()/len(data)
df_interSer=pd.DataFrame({'InterSer_no':y1,'DSL':y2,'Fiber optic':y3})
df_interSer.plot(kind='bar',stacked=True)
plt.xticks([0,1],['Retain','Churn'],rotation=1)
plt.title('InternetService')

plt.show()


#查看订购附加的在线安全服务，	订购附加的在线备份服务	，为公司提供的网络设备购买附加的设备保护服务，
#查看订购附加的技术支持以缩短等待时间，	是否使用第三方的流TV，是否使用第三方的流电影
#绘图
cols=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
fig=plt.figure(figsize=(18,15))
x=data['Churn'].unique()
for i,iterm in enumerate(cols):
    i+=1   
    y0=data.Churn[data[iterm]=='No'].value_counts()
    y1=data.Churn[data[iterm]=='Yes'].value_counts()
    y2=data.Churn[data[iterm]=='No internet service'].value_counts()
    df_iterm=pd.DataFrame({iterm+'_no':y0,iterm+'_yes':y1,'NoInterSer':y2})

    axi=fig.add_subplot(3,3,i)
    df_iterm.plot(kind='bar',stacked=True,ax=axi) 
    plt.xticks([0,1],['Retain','Churn'],rotation=1,fontsize=14)
    plt.title(iterm)

plt.show()

# x=data.tenure.unique()
# y0=data.tenure[data.Churn=='Yes'].value_counts()
# y1=data.tenure[data.Churn=='No'].value_counts()
# df_tenure=pd.DataFrame({'Churn':y0,'Retain':y1})
# df_tenure.plot(kind='bar',stacked=True,figsize=(20,4))
# plt.title('tenure')
# plt.show()




sns.regplot(x='tenure',y='TotalCharges',order=4,data=data)

## MonthlyCharges 和 TotalCharges 与流失的关系
s_value1 = list(data['TotalCharges'])
s_value2 = list(data.loc[(data['Churn'] == 'Yes'), 'TotalCharges'])
s_value3 = list(data.loc[(data['Churn'] == 'No'), 'TotalCharges'])
labels = ['num_all','num_yes','num_no']
fig = plt.figure(figsize=(12,6), facecolor='w')
plt.boxplot([s_value1, s_value2, s_value3], labels=labels, vert=False, showmeans=True)
plt.title('TotalCharges')

s_value1 = list(data['MonthlyCharges'])
s_value2 = list(data.loc[(data['Churn'] == 'Yes'), 'MonthlyCharges'])
s_value3 = list(data.loc[(data['Churn'] == 'No'), 'MonthlyCharges'])
labels = ['num_all','num_yes','num_no']
fig = plt.figure(figsize=(12,6), facecolor='w')
plt.boxplot([s_value1, s_value2, s_value3], labels=labels, vert=False, showmeans=True)
plt.title('MonthlyCharges')










#数据转换
binary_cols=['Dependents','Partner','PhoneService','PaperlessBilling','Churn']
for col in binary_cols:
    data[col]=data[col].map({'Yes':1,'No':0})
data.MultipleLines=data.MultipleLines.map({'Yes':2,'No':1,'No phone service':0})
data.PaymentMethod=data.PaymentMethod.map({'Credit card (automatic)':3,'Bank transfer (automatic)':2,'Mailed check':1,'Electronic check':0})
data.InternetService=data.InternetService.map({'No':0,'DSL':1,'Fiber optic':2})
data.gender=data.gender.map({'Female':0,'Male':1})
data.Contract=data.Contract.map({'Two year':2,'One year':1,'Month-to-month':0})
inter_cols=['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
for col in inter_cols:
    data[col]=data[col].map({'No':1,'Yes':2,'No internet service':0})




#重新检查列内值的分布情况，是否完成修改
col_number = ['customerID','tenure','MonthlyCharges','TotalCharges']
for col in data.columns.values:
    if col not in col_number:
        print('列名: {}\n{}\n{}\n'.format(col, '-'*20, data[col].value_counts()))


#删除customerID列
data.drop('customerID',axis=1,inplace=True)





#由于提供的数据集不包括数据的测试集，因此要从原始数据集中划分一部分作为测试数据
#划分训练集和测试集
y=data.Churn
X=data.drop(['Churn'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
#构造决策树
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
#模型评估
#利用classification_report()输出模型评估报告的方法。生成精准率，召回率，F1分数
#召回率（recall）：原本为对的当中，预测为对的比例（值越大越好，1为理想状态）
#精确率、精度（precision）：预测为对的当中，原本为对的比例（值越大越好，1为理想状态）
#F1分数（F1-Score）：综合了Precision与Recall的产出的结果（值越大越好，1为理想状态
predictdt_y = dt_model.predict(X_test)
accuracy_dt = dt_model.score(X_test,y_test)
print("决策树精确度 :",accuracy_dt)
print("决策树分类的评估结果")
print(classification_report(y_test, predictdt_y, digits=5))

# 可视化决策树并保存为图片
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dt_model, filled = True)
plt.savefig('decision_tree.png')
# 读取并显示图片
img = mpimg.imread('decision_tree.png')
imgplot = plt.imshow(img)
plt.show()


#集成学习梯度提升决策树
gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
print("梯度提升分类器", accuracy_score(y_test, gb_pred))
print(classification_report(y_test, gb_pred, digits=5))
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, gb_pred), annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("Gradient Boosting Classifier Confusion Matrix",fontsize=14)
plt.show()

#构造knn分类器
knn_model = KNeighborsClassifier(n_neighbors = 11) 
knn_model.fit(X_train,y_train)
#模型评估
predicted_y = knn_model.predict(X_test)
accuracy_knn = knn_model.score(X_test,y_test)
print("KNN精确度:",accuracy_knn)
print("KNN分类的评估结果")
print(classification_report(y_test, predicted_y, digits=5))

#逻辑回归
clf=LogisticRegression()
clf=clf.fit(X_train,y_train)
pred=clf.predict(X_train)
#使用交叉验证方法评估逻辑回归模型的性能，cv=10表示进行10折交叉验证。
score=cross_val_score(clf,X_test,y_test,cv=10)
#计算交叉验证的平均准确率。
valid_score=score.mean()
clf_pred= clf.predict(X_test)
print('逻辑回归精准度：',clf.score(X_test,y_test),'交叉验证集精准度：',valid_score)
print("逻辑回归的评估结果")
print(classification_report(y_test,clf_pred, digits=5))

#SVC
svc_model = SVC(random_state = 1)
svc_model.fit(X_train,y_train)
predict_y = svc_model.predict(X_test)
accuracy_svc = svc_model.score(X_test,y_test)
print("SVM 精确度 is :",accuracy_svc)


#随机森林
model_rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)
 
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))
print(classification_report(y_test, prediction_test, digits=5))
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, prediction_test),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("Random Forest Confusion Matrix ",fontsize=14)
plt.show()
y_rfpred_prob = model_rf.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_rfpred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_rf, tpr_rf, label='random forest',color = "r")
plt.xlabel('error rate')
plt.ylabel('real rate')
plt.title('Random Forest ROC Curve',fontsize=16)
plt.show();

#AdaBoost 分类器
a_model = AdaBoostClassifier()
a_model.fit(X_train,y_train)
a_preds = a_model.predict(X_test)
print("AdaBoost 分类器准确率")
metrics.accuracy_score(y_test, a_preds)
print(classification_report(y_test, a_preds, digits=5))
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, a_preds), annot=True,fmt = "d",linecolor="k",linewidths=3)
plt.title("AdaBoost Classifier Confusion Matrix",fontsize=14)
plt.show()

#投票分类器
clf1 = GradientBoostingClassifier()
clf2 = LogisticRegression()
clf3 = AdaBoostClassifier()
eclf1 = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
eclf1.fit(X_train, y_train)
predictions = eclf1.predict(X_test)
print("最终准确率得分 ")
print(accuracy_score(y_test, predictions))
plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, predictions),annot=True,fmt = "d",linecolor="k",linewidths=3)
 
plt.title("confusion matrix",fontsize=14)
plt.show()
print(classification_report(y_test, predictions, digits=5))

#线性判别分析
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
predictions = lda.predict(X_test)
accuracy_lda = lda.score(X_test,y_test)
print("Linear Discriminant Analysis accuracy is :",predictions)
print(classification_report(y_test, predictions, digits=5))





