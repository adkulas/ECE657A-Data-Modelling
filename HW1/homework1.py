import pandas as pd

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
df.columns = columns