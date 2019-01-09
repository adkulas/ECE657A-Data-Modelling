import pandas as pd
import seaborn as sns

df = pd.read_csv('data/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None)
columns = ['Sample_code_number',
           'clump_thickness',
           'Uniformity_of_Cell_Size',
           'Uniformity_of_Cell_Shape',
           'Marginal_Adhesion',
           'Single_Epithelial_Cell_Size',
           'Bare_Nuclei',
           'Bland_Chromatin',
           'Normal_Nucleoli',
           'Mitoses', 
           'Class']  
# df.set_index('Sample_code_number')
df.columns = columns

summary_stats = [df.mean(), df.skew(), df.std(), df.var()]
results_df = pd.DataFrame()
results_df['mean'] = df.mean()
results_df['skew'] = df.skew()
results_df['std_dev'] = df.std()
results_df['variance'] = df.var()

results_df = results_df.T
print(results_df.head())

correlations = df.corr(method='pearson')
print(correlations.head(100))

benign = df.loc[df['Class'] == 2]
malignant = df.loc[df['Class'] == 4]
print(benign.head())

sns.distplot(benign.iloc[:,2])