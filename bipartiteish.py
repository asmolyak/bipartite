#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 10:51:32 2016

@author: ianir
"""
from __future__ import division
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

start = datetime.now()

def init_values(num_of_assets,num_of_banks,min_price = 10, max_price = 100, min_holding = 2, max_holding = 20, with_other = True,holding_probability = 0.5):
    num_of_assets = num_of_assets+with_other
    prices = np.asfarray(np.random.randint(min_price, max_price,num_of_assets))
    num_of_holders = num_of_banks+with_other
    holdings = np.zeros([num_of_holders,num_of_assets])
    for i in range(num_of_holders):
        holdings[i,:] = np.random.randint(min_holding,max_holding,num_of_assets)*np.random.binomial(1,holding_probability,num_of_assets)
    return prices, holdings


def init_assets(prices,num_of_assets,holdings):
    assets = {}
    init_value = prices*sum(holdings)
    min_value = 0.8*np.random.rand(len(prices))*prices
    for i in range(num_of_assets):
        assets['a'+str(i)]=(prices[i],init_value[i])
    return assets, min_value
    
def init_banks(holdings,prices,num_of_banks,min_liab = 0.75,max_liab = 0.95,with_other = True):
    banks = {}
    init_value = np.dot(holdings,prices)
    liabilities = ((max_liab-min_liab)*np.random.rand(num_of_banks+with_other)+min_liab)*init_value
    for i in range(num_of_banks):
        banks['b'+str(i)] = init_value[i]
    return banks, liabilities

def init_network(banks,assets, holdings,internals = False,num_of_internals = 100,type_of_conns = ['partner']):
    G = nx.Graph()
    for key_banks, value_banks in banks.iteritems():
        bank_index = int(key_banks[1:])
        G.add_node(key_banks,attr_dict = {'kind':'bank','value':value_banks})
        for key_assets, value_assets in assets.iteritems():
            asset_index = int(key_assets[1:])            
            if key_assets not in G:
                G.add_node(key_assets,attr_dict = {'kind':'asset','price':value_assets[0],'value':value_assets[1]})
            if holdings[bank_index,asset_index]>0:
                G.add_edge(key_banks,key_assets,attr_dict={'holdings':holdings[bank_index,asset_index]})
    if internals:
        for i in range(num_of_internals):
            nodes_to_attach = np.random.choice(num_of_assets,2)
            G.add_edge('a'+str(nodes_to_attach[0]),'a'+str(nodes_to_attach[1]),attr_dict = {'kind':type_of_conns[np.random.choice(len(type_of_conns))],'strength':1})
    return G
        
def hit_network(G,prices,prob_to_hit=0.2,hit=0.8):
    for asset, data in G.nodes_iter(data = True):
        if data['kind'] == 'asset' and np.random.rand()<prob_to_hit:
            data['value']*=hit
            data['price']*=hit
            prices[int(asset[1:])]*=hit
    return G, prices

def time_step():
    pass

def eval_network(G,holdings,prices,min_value,liabilities,impact,failure_in='bank'):
    node_to_check = [1]
    inner_loop=0
    while len(node_to_check)>0:
        inner_loop += 1
        node_to_check = []
        if failure_in == 'bank':
            new_bank_value = np.dot(holdings,prices)
            for node,data in G.nodes_iter(data = True):
                if data['kind'] == 'bank' and data['value']>0:
                    ind = int(node[1:])
                    if new_bank_value[ind] < liabilities[ind]:
                        data['value'] = 0
                        holdings[ind,:] = 0
                        node_to_check.append(node)
                    else:
                        data['value'] = new_bank_value[ind]
            for node in node_to_check:
                for neigh_node in G.neighbors_iter(node):
                    G.node[neigh_node]['price']*=(1-impact)
                    prices[int(neigh_node[1:])]*=(1-impact)
        elif failure_in == 'asset':
            new_asset_value = prices*sum(holdings)
            for node,data in G.nodes_iter(data=True):
                if data['kind']=='asset' and data['value']>0:
                    ind = int(node[1:])
                    if new_asset_value[ind] < min_value[ind]:
                        data['value'] = 0
                        holdings[:,ind] = 0
                        node_to_check.append(node)
                    else:
                        data['value'] = new_asset_value[ind]
            for node in node_to_check:
                for neighbors in G.neighbors_iter(node):
                    if G.node[neighbors]['kind']=='bank':
                        for asset_neighbors in G.neighbors_iter(neighbors):
                            G.node[asset_neighbors]['price']*=(1-impact)
                            prices[int(asset_neighbors[1:])]*=(1-impact)
                    else:
                        if G.get_edge_data(node,neighbors)['kind'] == 'competitor':
                            G.node[neighbors]['price']/=(0.68*(1-impact)/(G.get_edge_data(node,neighbors)['strength']))
                            prices[int(neighbors[1:])]/=(0.68*(1-impact)/(G.get_edge_data(node,neighbors)['strength']))
                        elif G.get_edge_data(node,neighbors)['kind'] == 'partner':
                            G.node[neighbors]['price']*=(0.68*(1-impact)/(G.get_edge_data(node,neighbors)['strength']))
                            prices[int(neighbors[1:])]*=(0.68*(1-impact)/(G.get_edge_data(node,neighbors)['strength']))
    print inner_loop
    return G,prices,holdings,inner_loop
#some params    
num_of_banks = 150
num_of_assets = 7000
assets_per_bank = 100
#Inits
prices, holdings = init_values(num_of_assets=num_of_assets, num_of_banks=num_of_banks,holding_probability=assets_per_bank/num_of_assets, with_other=False)
assets, min_asset_price = init_assets(prices=prices,num_of_assets=num_of_assets,holdings= holdings)
banks, liabilities = init_banks(holdings,prices,num_of_banks, with_other=False)
a = init_network(banks=banks,assets=assets,holdings=holdings,internals=True,num_of_internals=6000)

#loop backups
copy_a = a.copy()
copy_prices = prices.copy()
hold_cop = holdings.copy()
container = {}

for imp in np.arange(0.0001,0.01,0.0001):
    a = copy_a.copy()
    holdings = hold_cop.copy()
    prices = copy_prices.copy()
    a, prices = hit_network(G=a,prices=prices,prob_to_hit=0.1,hit=0.79)
    hold_cop = holdings.copy()
    a, prices, holdings,num_of_iters = eval_network(a,holdings,prices,min_asset_price*sum(holdings),liabilities=liabilities,impact=imp,failure_in='asset')
    print imp, sum(np.sum(hold_cop,0)>0),sum(np.sum(holdings,0)>0)
    container[imp] = (num_of_iters,sum(sum(holdings>0)))
print datetime.now() - start
pd.DataFrame(container,index=['iters','sum']).T['iters'].plot(style='-.',legend = True)
pd.DataFrame(container,index=['iters','sum']).T['sum'].plot(secondary_y=True,style='.',legend='True')