import pandas as pd

dataset = pd.read_csv(r'D:\#PROJECTS\Road Accident Severity Prediction System\Road.csv\Road.csv')
df = pd.DataFrame(dataset)

print(df.isnull().sum())

print(df.isnull().sum().sum()) # 20057

print(dataset.shape) #((12316, 32))


total_values = df.size # 12316*32 = 394112
#it Includes nulls and non null values both
print(total_values)


total_values = 394112
null_values = 20057


df.dropna(inplace=True)
print(df.shape) #(2889, 32) -> shape after droping


total_rows = 12316
rows_pending = 2889


#data loss by drop na (in %)
data_loss = ((total_rows - rows_pending)/total_rows)*100

data_loss = round(data_loss, 2)
print(f"data lossed :{data_loss}%")
#                        76.54%                        #