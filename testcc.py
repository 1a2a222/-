import pandas as pd

list=[[1,2,3],[1,3,3],[1,2,4]]
list=[[[[1,2,3]]]]
name = ['1','2','3']
test = pd.DataFrame(data=list)
test.to_csv('D:/xdata1.csv')