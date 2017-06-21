#!/usr/bin/env python2
# -*- coding: utf-8 -*-


from __future__ import division
from scipy.stats import powerlaw
from scipy import optimize
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib import cm
import pandas as pd
from collections import defaultdict
import pickle
from datetime import date

def draww(X,Y,Z,xlabel,ylabel,zlabel,title):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
#    X=hit_range
#    Y = health_threshold
    X,Y = np.meshgrid(X,Y)
    surf = ax.plot_surface(X,Y,Z.as_matrix().T,cmap=cm.coolwarm,rstride=1,cstride=1,linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def average_degree(graph,size=False):
    if size>0:
        return sum([j for (i,j) in graph.degree_iter()])/float(size)
    else:
        return sum([j for (i,j) in graph.degree_iter()])/float(len(graph))

def calc_min_holdings(average,power):
    return average*(power-1)/(power)


def generate_distribution(p,min_holdings,num_of_banks,num_of_assets,fit=False,fixed_mean = False, average = 16+2.0/3):
    if fixed_mean:
        min_holdings = calc_min_holdings(average,p)
    while True:
        rvss = 1/powerlaw.rvs(p,scale=1/min_holdings,size=num_of_banks)
        if max(rvss)<num_of_assets:
            break
    fixed_rvs = rvss.copy()
    round_rvs = map(round,fixed_rvs)
    print np.mean(round_rvs)
    if fit:
        hist, bins= np.histogram(rvss, bins =np.logspace(np.log10(min(rvss)),np.log10(max(rvss))))
        fitfunc = lambda p,x: p[0]*x**p[1]
        errfunc = lambda p,x,y: fitfunc(p,x) - y
        p0theo = (p-1)*(min_holdings**(p-1))
        p0 = [0.5,-3]
        p1,success = optimize.leastsq(errfunc,p0[:], args = (bins[:-1],hist))
#        plt.plot(bins[:-1],hist,'.',hold=True)
#        plt.hold()
#        plt.plot(bins[:-1],fitfunc(p1,bins[:-1]),hold=True)
        print p1, p0theo, np.mean(rvss)
    return round_rvs

def setup_network(num_of_banks,num_of_assets,round_rvs,kind='man_half_half',p=2.5,min_holdings=10,single_net = False,project_network=False,av_deg = False,add_fedfunds=None):
    if not single_net:
        if kind == 'er':
            if av_deg:
                G=nx.bipartite.random_graph(num_of_banks,num_of_assets,av_deg/num_of_banks)
            else:
                G=nx.bipartite.random_graph(num_of_banks,num_of_assets,np.mean(round_rvs)/num_of_banks)
            bot,top = nx.bipartite.sets(G)
            assets = {t:'a' +str(t-num_of_banks) for t in top }
            banks = {b:'b' +str(b) for b in bot}
            z = assets.copy()
            z.update(banks)
            nx.relabel_nodes(G,z,copy=False)
            for node,data in G.nodes_iter(data=True):
                if node.startswith('b'):
                    data['kind'] = 'bank'
                else:
                    data['kind'] = 'asset'
            print 'random: ', average_degree(G)
        elif kind == 'powpow':
            G=nx.bipartite.configuration_model(map(int,round_rvs),map(int,round_rvs),create_using = nx.Graph())
            bot,top = nx.bipartite.sets(G)
            assets = {t:'a' +str(t-num_of_banks) for t in top }
            banks = {b:'b' +str(b) for b in bot}
            z = assets.copy()
            z.update(banks)
            nx.relabel_nodes(G,z,copy=False)
            for node,data in G.nodes_iter(data=True):
                if node.startswith('b'):
                    data['kind'] = 'bank'
                else:
                    data['kind'] = 'asset'
            print 'powpow: ', average_degree(G)
        elif kind == 'man_half_half':
            banks = ['b'+str(i) for i in range(num_of_banks)]
            assets = ['a'+str(i) for i in range(num_of_assets)]
            G = nx.Graph()
            G.add_nodes_from(banks,kind='bank')
            G.add_nodes_from(assets,kind='asset')
            for i,j in enumerate(round_rvs):
                G.add_edges_from([('b'+str(i),'a'+str(k)) for k in np.random.choice(num_of_assets,int(j),replace=False)])
            print 'power law: ', average_degree(G)
        elif kind == 'auto_half_and_half':
            temp_rvs = round_rvs[:-1]        
            while True:            
                pois = np.random.poisson(np.mean(round_rvs),num_of_assets)
                pd.Series(pois).hist()
                rem = sum(pois)-sum(temp_rvs)
                if rem < 0:
                    continue
                else:
                    if ((p-1)*(min_holdings**(p-1)))*(rem**(-p)) > 0.0001:
                        round_rvs[-1] = rem
                        break
                    else:
                        continue
            G = nx.bipartite.configuration_model(map(int,round_rvs),map(int,pois),create_using = nx.Graph())
            bot,top = nx.bipartite.sets(G)
            assets = {t:'a' +str(t-num_of_banks) for t in top }
            banks = {b:'b' +str(b) for b in bot}
            z = assets.copy()
            z.update(banks)
            nx.relabel_nodes(G,z,copy=False)
            for node,data in G.nodes_iter(data=True):
                if node.startswith('b'):
                    data['kind'] = 'bank'
                else:
                    data['kind'] = 'asset'
            print 'auto_h_h: ', average_degree(G)
        if project_network:
            G = nx.bipartite.project(G,[node for node,data in G.nodes_iter(data=True) if data['kind']=='bank'])
            for node,data in G.nodes_iter(data=True):
                data['init_deg'] = G.degree(node)
        else:
            if add_fedfunds:
                G.add_node('f01')
                new_edges = [('f01','b'+str(i)) for i in random.sample(range(num_of_banks),add_fedfunds)]
                G.add_edges_from(new_edges)
                G.node['f01']['kind']='fed'                

    else:
        if kind=='er':
            G = nx.fast_gnp_random_graph(num_of_banks,np.mean(round_rvs)/num_of_banks)
        elif kind=='man_half_half':
            if not nx.is_valid_degree_sequence(map(int,round_rvs)):
                round_rvs[0]+=1
            G = nx.configuration_model(map(int,round_rvs),create_using=nx.Graph())
        banks = {b:'b' +str(b) for b in range(num_of_banks)}
        nx.relabel_nodes(G,banks,copy=False)
        for node, data in G.nodes_iter(data=True):
            data['kind'] = 'bank'
    for node,data in G.nodes_iter(data=True):
        data['init_deg'] = G.degree(node)
    return G


def run_core(params,hit_probability=0.005):
#    fig = plt.figure()
#    gather_gc_by_avdeg = fig.gca()
#    fig2 = plt.figure()
#    gather_iters_by_avdeg = fig2.gca()
    fract_threshold,kind = params[0], params[1]
    num_of_banks = 10000
    num_of_assets = 10000
    if fract_threshold:
        health_threshold = np.arange(0,1,0.0001)
    else:
        health_threshold = np.arange(0,10,.001)#, th_base/((1+th_base)**fract_threshold)#np.arange(0,1,0.01)#
#        hit_range = [0.001]#np.arange(0,.01,0.0001)
    av_deg_range = np.arange(2,10)#np.logspace(np.log10(0.01),np.log10(18),100)
    power = 2.5
    min_holdings = 5

    network_health = pd.DataFrame(columns = av_deg_range,index = health_threshold)
    iterations = network_health.copy()
    degree = network_health.copy()
    max_clust = network_health.copy()
    core_num = network_health.copy()
    bank_nodes = network_health.copy()
    asset_ndoes = network_health.copy()
    fail_counter = pd.DataFrame()
    size_counter = pd.DataFrame()
    for avg in av_deg_range:
        print 'running avg degg', avg
        round_rvs = generate_distribution(power,min_holdings,num_of_banks=num_of_banks,num_of_assets=num_of_assets,fit=False,fixed_mean=True,average=avg)
        single_net = False
        project_network = False
        link_temp = 0
    
        G = setup_network(num_of_banks=num_of_banks,num_of_assets=num_of_assets,round_rvs=round_rvs,p=power,kind=kind,min_holdings=min_holdings,\
                          single_net=single_net,project_network=project_network,av_deg=avg)
        output = {}
        G_copy = G.copy()
        for th in health_threshold:
            local_counter = []
            local_size = []
            G = G_copy.copy()
            if project_network or single_net:
                init_nodes_to_remove = ['b'+str(i) for i in random.sample(range(num_of_assets),int(hit_probability*num_of_assets))]
            else:
                init_nodes_to_remove = ['a'+str(i) for i in random.sample(range(num_of_assets),int(hit_probability*num_of_assets))]
            link_temp = sum(G.degree(init_nodes_to_remove).values())
            G.remove_nodes_from(init_nodes_to_remove)
            local_counter.append(len(init_nodes_to_remove))
            counter = 0
            while True:
                counter +=1
                b_nodes_to_remove = set([node for node,data in G.nodes_iter(data=True) if \
                                         data['kind']=='bank' and (data['init_deg']==0 or G.degree(node)/(data['init_deg']**fract_threshold)<=th  or \
                                             (not fract_threshold)*(G.degree(node)<=(th+1))*(np.random.rand()<(th%1)))])#
                if len(b_nodes_to_remove) == 0:
                    break
                local_counter.append(len(b_nodes_to_remove))
                link_temp += sum(G.degree(b_nodes_to_remove).values())
                G.remove_nodes_from(b_nodes_to_remove)
                try:
                    sz = len(max(nx.connected_components(G),key=len))
                except:
                    sz = 0
                local_size.append(sz)

                if not project_network:
                    a_nodes_to_remove = set([node for node,data in G.nodes_iter(data=True) if \
                                             data['kind']=='asset' and (data['init_deg']==0 or G.degree(node)/(data['init_deg']**fract_threshold)<=th or \
                                                 (not fract_threshold)*(G.degree(node)<=(th+1))*(np.random.rand()<(th%1)))])#
                    if len(a_nodes_to_remove) == 0:
                        break
                    link_temp += sum(G.degree(a_nodes_to_remove).values())
                    G.remove_nodes_from(a_nodes_to_remove)
            try:
                temp = average_degree(G,num_of_assets+num_of_banks)
                temp2 = len(max(nx.connected_components(G),key = len))/(num_of_assets+num_of_banks)#len(G)
                temp3 = max(nx.core_number(G).values())
            except:
                temp = 0
                temp2 = 0
                temp3 = 0
            temp4 = len([node for node, data in G.nodes_iter(data = True) if data['kind']=='bank'])
            temp5 = len([node for node, data in G.nodes_iter(data = True) if data['kind']=='asset'])    
            fail_counter = pd.concat([fail_counter,pd.Series(local_counter,name=(int(avg),th))],axis=1)
            size_counter = pd.concat([size_counter,pd.Series(local_size,name=(int(avg),th))],axis=1)
            output[th] = (counter,len(G),temp,temp2,temp3,temp4,temp5)
        print 'done iterating'
        df = pd.DataFrame(output).T
        network_health[avg] = df[1]
        iterations[avg]=df[0]
        degree[avg] = df[2]
        max_clust[avg] = df[3]
        core_num[avg] = df[4]
        bank_nodes[avg] = df[5]
        asset_ndoes[avg] = df[6]
    print 'done building df'
    print 'all done'
    return {(kind,fract_threshold):(network_health,iterations,degree,max_clust,fail_counter,size_counter)}

if __name__ == "__main__":
    from multiprocessing.pool import Pool
    print "let's give it a go"
    fract_threshold = [True,False]
    kind = ['er','powpow','man_half_half']#
    params = [(u,v) for u in fract_threshold for v in kind]
    pool = Pool()
    results = pool.map(run_core,params)
    pool.close()
    pool.join()
    with open('results'+str(date.today())+'.pickle', 'wb') as h:
        pickle.dump(results,h,protocol=pickle.HIGHEST_PROTOCOL)
    print 'done with pool'
    for i in results:
        for k,v in i.iteritems():
            print k
            x=[]
            y=[]
            for col in v[2].columns:
                x.append(col)
                y.append( v[2][col][v[2][col]>0.1].min())
            plt.plot(x,y)
