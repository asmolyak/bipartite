# -*- coding: utf-8 -*-
"""
Created on Wed Mar 01 13:15:00 2017

@author: Alex
"""

from __future__ import division
import random
import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

'''
analysis of bank-sov data set
'''

def average_degree(graph):
    return sum(graph.degree().values())/len(graph)

def set_init(graph):
    for node, data in graph.nodes(data=True):
        if len(node) > 2:
            data['kind'] = 'bank'
        else:
            data['kind'] = 'asset'
        data['init_value'] = float(sum([a['weight'] for a in graph[node].values() if type(a)==dict]))
        data['value'] = data['init_value']
        data['any_fragile_neigh']={}
        data['all_fragile_neigh']={}
        data['avg_fragile_neigh']={}
        data['med_fragile_neigh']={}
        data['num_of_frag_neigh']={}
    return graph


data = pd.read_excel('d:/Downloads/EU_SOV_Debt_Bank_Net_2011.xlsx',header=0,index_col=0)
data2 = pd.read_excel('d:/Downloads/Credit risk exposures and exposures to sovereigns.xlsx',header=1,index_col=0)
data2_clean = data2[data2.columns[[0,1,2,3,7]]]
norm_factor = data/data.sum()

data_sovn = data2_clean[data2_clean.columns[0]]*norm_factor
data_fin = data2_clean[data2_clean.columns[1]]*norm_factor
data_corp = data2_clean[data2_clean.columns[2]]*norm_factor
data_retail = data2_clean[data2_clean.columns[3]]*norm_factor
data_realest = data2_clean[data2_clean.columns[4]]*norm_factor
data_tot = data_sovn + data_fin + data_corp + data_retail + data_realest


sovereign_debt = nx.bipartite.from_biadjacency_matrix(sp.sparse.csr_matrix(data_sovn))
fin_debt = nx.bipartite.from_biadjacency_matrix(sp.sparse.csr_matrix(data_fin))
corp_debt = nx.bipartite.from_biadjacency_matrix(sp.sparse.csr_matrix(data_corp))
retail_debt = nx.bipartite.from_biadjacency_matrix(sp.sparse.csr_matrix(data_retail))
realest_debt = nx.bipartite.from_biadjacency_matrix(sp.sparse.csr_matrix(data_realest))
tot_debt = nx.bipartite.from_biadjacency_matrix(sp.sparse.csr_matrix(data_sovn + data_fin + data_corp + data_retail + data_realest))

mapping = {i:j for i,j in zip(sovereign_debt.nodes(),data.index.append(data.columns))}
sovereign_debt=nx.relabel_nodes(sovereign_debt,mapping)
fin_debt=nx.relabel_nodes(fin_debt,mapping)
corp_debt=nx.relabel_nodes(corp_debt,mapping)
retail_debt=nx.relabel_nodes(retail_debt,mapping)
realest_debt=nx.relabel_nodes(realest_debt,mapping)
tot_debt=nx.relabel_nodes(tot_debt,mapping)

sovereign_debt = set_init(sovereign_debt)
fin_debt = set_init(fin_debt)
corp_debt = set_init(corp_debt)
retail_debt = set_init(retail_debt)
realest_debt = set_init(realest_debt)
tot_debt = set_init(tot_debt)

all_assets = [sovereign_debt,fin_debt,corp_debt,retail_debt,realest_debt]

top, bottom = nx.bipartite.sets(sovereign_debt)
#print top, '\n\n' , bottom
#print average_degree(sovereign_debt),len(top), sum(sovereign_debt.degree(top).values())/len(top),len(bottom),sum(sovereign_debt.degree(bottom).values())/len(bottom)
#pd.Series(sovereign_debt.degree(top).values()).hist(bins=10,alpha=0.5)
#pd.Series(sovereign_debt.degree(bottom).values()).hist(bins=5,alpha=0.5)

#randr = nx.bipartite.random_graph(32,90,0.39791666666666664)
#print nx.bipartite.average_clustering(sovereign_debt,mode='dot'), nx.bipartite.average_clustering(randr,mode='dot')
sovereign_debt_copy = sovereign_debt.copy()

'''
Figure 5 for paper - data visualisation


def get_cmap(n, name='tab20'):
#    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
#    RGB color; the keyword argument name must be a standard mpl colormap name.
    return plt.cm.get_cmap(name, n)

doms = {j[:2] for j in bottom}
clrs = {}
for n,d in enumerate(doms):
    clrs[d] = n
cmap = get_cmap(len(doms))
x=[];y=[];size=[];color=[]
for i in bottom:
    x.append(sovereign_debt_copy.node[i]['init_value'])
    y.append(np.random.rand())
    size.append(sovereign_debt_copy.degree(i))
    color.append(cmap(clrs[i[:2]]))
# fig 5a 
fig = plt.figure(figsize=(16,12))
plt.scatter(x,y,s=[(kr*50) for kr in size],c=color)
plt.legend([mpatches.Patch(color=cmap(b)) for b in clrs.values()],
           [i for i in clrs.keys()])
plt.xlabel('Overall Exposure',fontsize=20)
plt.yticks([])


# fig 5b 
fig = plt.figure(figsize=(16,12))
plt.tick_params(labelsize=20)
plt.scatter(x,y,s=[(kr*50) for kr in size],c=color)
plt.semilogx(True)
plt.legend([mpatches.Patch(color=cmap(b)) for b in clrs.values()],
           [i for i in clrs.keys()])
plt.xlabel('Overall Exposure (Log scale)',fontsize=20)
plt.yticks([])

#fig 5c number of banks per country

count = {}; exposure = {}
for i in bottom:
    if i[:2] in count.keys():
        count[i[:2]] += 1
        exposure[i[:2]] += sovereign_debt_copy.node[i]['init_value']
    else:
        count[i[:2]] = 1
        exposure[i[:2]] = sovereign_debt_copy.node[i]['init_value']

fig = plt.figure(figsize=(16,12))
plt.tick_params(labelsize=20)
sr = pd.Series(count)
sr.sort_values(ascending=False,inplace=True)
sr.plot(kind='bar',color=[cmap(clrs[k[:2]]) for k in sr.index])
plt.ylabel('Number of Banks',fontsize=20)
plt.xlabel('Country',fontsize=20)


fig = plt.figure(figsize=(16,12))
plt.tick_params(labelsize=20)
exp = pd.Series(exposure)
exp.sort_values(ascending=False,inplace=True)
exp.plot(kind='bar',color=[cmap(clrs[k[:2]]) for k in exp.index],logy=True)
plt.ylabel('Exposure',fontsize=20)
plt.xlabel('Country',fontsize=20)



end fig 5
'''


#any_fragile_neigh = {}
#all_fragile_neigh = {}
#avg_fragile_neigh = {}
#med_fragile_neigh = {}
#num_of_frag_neigh = {}
rg = np.arange(0.17,0.02,-0.01)
out = pd.DataFrame(index = top, columns = rg)
outbanks = pd.DataFrame(index = top, columns =rg)
tot_size = pd.DataFrame(len(sovereign_debt),index = top, columns =rg)
tot_banks = pd.DataFrame(len(bottom),index = top, columns =rg)
saved_banks = pd.DataFrame(index = top, columns =rg)
immunized_banks = pd.DataFrame(index = top, columns =rg)

safe_nodes = {}
safe_perc = {}
tmp = {}
safe_banks = {}
for threshold in rg:
    safe_nodes[threshold] = []
    safe_banks[threshold] = []
    sovereign_debt = sovereign_debt_copy.copy()
    top, bottom = nx.bipartite.sets(sovereign_debt)
    for k,d in sovereign_debt.nodes(data=True):
        d['any_fragile_neigh'][threshold] = [any([(w['weight']/d['init_value'])>threshold for _,_,w in sovereign_debt.edges(k,data=True)])]
        d['all_fragile_neigh'][threshold] = [all([(w['weight']/d['init_value'])>threshold for _,_,w in sovereign_debt.edges(k,data=True)])]
        d['avg_fragile_neigh'][threshold] = [np.mean([(w['weight']/d['init_value']) for _,_,w in sovereign_debt.edges(k,data=True)])>threshold]
        d['med_fragile_neigh'][threshold] = [np.median([(1-w['weight']/d['init_value']) for _,_,w in sovereign_debt.edges(k,data=True)])>threshold]
        d['num_of_frag_neigh'][threshold] = [sum([(sovereign_debt.edges()[(k, node2)]['weight'] / sovereign_debt.node[node2]['init_value'])>threshold for node2 in 
         sovereign_debt.neighbors(k)])]
        if (d['num_of_frag_neigh'][threshold][0]>1 and d['any_fragile_neigh'][threshold][0]==True):
            safe_nodes[threshold].append(k)
            if len(k)>=2:
                safe_banks[threshold].append(k)
#            print threshold, top
    broadg = nx.subgraph(sovereign_debt, [n for n,d in sovereign_debt.nodes(data=True) if (d['any_fragile_neigh'][threshold][0]*d['num_of_frag_neigh'][threshold][0])>=1 ])
    avg = np.mean(dict(broadg.degree()).values())
    safe_nodes[threshold] = []#safe_banks[threshold]#[n for n, d in broadg.degree() if d>=(avg-1)] #

    for init_hit in top:#= random.sample(top,1)[0]
        sovereign_debt = sovereign_debt_copy.copy()
        top, bottom = nx.bipartite.sets(sovereign_debt)
        banks_to_update = set([node for node in sovereign_debt[init_hit] if (node not in safe_nodes[threshold]) and
                               (sovereign_debt[node][init_hit]['weight']/sovereign_debt.node[node]['init_value']>threshold)])# 
#        print 'init: ', threshold, banks_to_update    
        sovereign_debt.remove_node(init_hit)
        for node, data in sovereign_debt.nodes(data=True):
            data['value'] = sum([a['weight'] for a in sovereign_debt[node].values() if type(a)==dict])
#            print node, data['init_value'], data['value']
        removed_banks = []
        removed_sovs = []
        while True:
            countries_to_update = set()
            while len(banks_to_update) > 0:
                bank = banks_to_update.pop()
                countries_to_update |= set([node for node in sovereign_debt[bank] if sovereign_debt[node][bank]['weight']/sovereign_debt.node[node]['value']>threshold])
#                print threshold, countries_to_update
                removed_banks.append(bank)
                sovereign_debt.remove_node(bank)
                for node, data in sovereign_debt.nodes(data=True):
                    data['value'] = sum([a['weight'] for a in sovereign_debt[node].values() if type(a)==dict])
#                    print 'in banks', threshold, node, data['init_value'], data['value']

            if len(countries_to_update) == 0:
                break
            banks_to_update = set()
            while len(countries_to_update) > 0:
                country = countries_to_update.pop()
                banks_to_update |= set([node for node in sovereign_debt[country] if  (node not in safe_nodes[threshold]) and 
                                        (sovereign_debt[node][country]['weight']/sovereign_debt.node[node]['value']>threshold)])#
                removed_sovs.append(country)
#                print threshold, banks_to_update
                sovereign_debt.remove_node(country)
                for node, data in sovereign_debt.nodes(data=True):
                    data['value'] = sum([a['weight'] for a in sovereign_debt[node].values() if type(a)==dict])
#                    print 'in countries', threshold, node, data['init_value'], data['value']
            if len(banks_to_update) == 0:
                break
#        top, bottom = nx.bipartite.sets(sovereign_debt)
    #    if not (nx.is_connected(sovereign_debt)):
        if (len(removed_sovs)+len(removed_banks))>0 :
            if np.isclose(threshold,0.1):
                tmp[init_hit] = sovereign_debt.nodes()
            out[threshold][init_hit] = len(removed_sovs)
            outbanks[threshold][init_hit] = len(removed_banks)
            tot_size[threshold][init_hit] = len(sovereign_debt)
            tot_banks[threshold][init_hit] =  len(bottom)
            immunized_banks[threshold][init_hit] = len([b for b in safe_nodes[threshold] if len(b)>2])
            saved_banks[threshold][init_hit] =  len(bottom) - len([b for b in safe_nodes[threshold] if len(b)>2])
            
#            print len(sovereign_debt)   
#fig = plt.figure(6,figsize=(16,12))
#plt.tick_params(labelsize=20)    
text_size=30
tot_size[tot_size.columns[3:-1:4]].plot(kind='bar',width=.8,figsize=(16,12))
plt.rc('xtick', labelsize=text_size)
plt.rc('ytick', labelsize=text_size)
plt.rc('legend', fontsize=text_size)
plt.rc('axes', labelsize=text_size)

plt.xlabel('Initial Failed Sovereign, Protected',fontsize=text_size)
plt.ylabel('Surviving nodes',fontsize=text_size)

fig = plt.figure(6,figsize=(16,12))
plt.tick_params(labelsize=text_size)
ax = immunized_banks.loc['DE'].plot(fontsize=text_size)
plt.xlabel('1 - Threshold',fontsize=text_size)
plt.ylabel('Number of Banks')


