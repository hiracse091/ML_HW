import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

df = pd.read_csv('data/task5.csv')
#print(df)

from scipy.stats import entropy

def generate_entropy(n):
    col0 = n.loc[n['A'] == 0]['A'].count()
    col1 = n.loc[n['A'] == 1]['A'].count()
    return entropy([col0, col1], base = 2)

def classify_data(d, c):
    n1 = d.loc[d[c] == 0]
    ent_n1 = generate_entropy(n1)
    
    n2 = d.loc[d[c] == 1]
    ent_n2 = generate_entropy(n2)
    total_entropy = ent_n1 * n1['A'].count() /d['A'].count() + ent_n2 * n2['A'].count() /d['A'].count()
    return total_entropy
    
    
def select_node(d):
    gain_max = 0
    selected = ''
    base_entropy = generate_entropy(d)
    for c in d.columns:
        if c != 'A':
            ent = classify_data(d, c)

            gain = base_entropy - ent

            if gain > gain_max:
                gain_max = gain

                selected = c
    return selected, gain, ent

# for depth 1 we select one node


def second_node(d):
    c, gain, ent = select_node(d)
    n1 = d.loc[d[c] == 1]
    n1_c,_,ent_n1 = select_node(n1)
    n2 = d.loc[d[c]== 0]
    n2_c,_,ent_n2 = select_node(n2)
    total_entropy = ent_n1 * n1['A'].count() / d['A'].count() + ent_n2 * n2['A'].count() / d['A'].count()
    
    
    gain = ent- total_entropy
    
    
    return n1_c, n2_c
    
    
second_node(df)


def decision_tree(d,max_depth=1):
    error = 0
    if max_depth == 1:
        for i,val in df.iterrows():
            selected, gain, ent = select_node(d)
            cls = d.loc[d[selected] == val[selected]]['A'].median() 
            if cls != val['A']:
                error += 1
    else:
        for i,val in df.iterrows():
            c, gain, ent = select_node(d)
            n1_c, n2_c = second_node(d)   
            if val[c] == 0:
                ds = d.loc[d[c] == 0]
                cls = ds.loc[ ds[n2_c] == val[n2_c] ]['A'].median()
                if cls != val['A']:
                    error += 1
            else:
                ds = d.loc[d[c] == 1]
                
            
                cls = ds.loc[ ds[n1_c] == val[n1_c] ]['A'].median()
                if int(cls) != val['A']:
                    error += 1
            
                                              
    return error/len(d)
    
error_max_depth_1 = decision_tree(df,1)
error_max_depth_2 = decision_tree(df,2)


print("Error MAX Depth 1", error_max_depth_1)
print("Error MAX Depth 2", error_max_depth_2)
