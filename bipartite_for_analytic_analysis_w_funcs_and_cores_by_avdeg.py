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

def paper_plot(list_of_points,reverse = False):
    fig, axes = plt.subplots(2,sharex=True)
    if reverse:
        if len(list_of_points)==0:
            network_health[::2].plot(ax=axes[0],legend=False)
            iterations[::2].plot(ax=axes[1],legend=False)
        network_health.ix[list_of_points].T[0.3:0.7].plot(ax=axes[0])
        iterations.ix[list_of_points].T[0.3:0.7].plot(ax=axes[1])
        axes[0].set_xlabel('Surviving nodes per threshold size for various impacts')
        axes[1].set_xlabel('Threshold')
    else:
        if len(list_of_points)==0:
            network_health[::5].plot(ax=axes[0],legend=False)
            iterations[::5].plot(ax=axes[1],legend=False)
        else:
            try:
                network_health[list_of_points].plot(ax=axes[0])
                iterations[list_of_points].plot(ax=axes[1])
            except:
                list_of_points = [l for l in list_of_points if l in network_health.index]
                network_health[list_of_points].plot(ax=axes[0])
                iterations[list_of_points].plot(ax=axes[1])

        axes[0].set_xlabel('Surviving nodes per Impact size for various thresholds')        
        axes[1].set_xlabel('Impact size')
    if len(list_of_points)>0:
        axes[0].legend(fontsize='xx-small')    
        axes[1].legend(fontsize='xx-small')    
    
    axes[0].set_ylabel('Number of Nodes')
    axes[1].set_ylabel('Iterations')
    axes[0].set_title('Surviving nodes and counts \npower=%.1f, MinDegree=%d, Kind=%s'%(power,min_holdings,kind))


def draw(X,Y,Z,xlabel,ylabel,zlabel,title):
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


def frag_centrality(G,th):
    out = {n:sum([G.degree(i)<(1/(1-th)) for i in G.neighbors_iter(n)]) for n in G.nodes_iter() if n.startswith('b')}
    return out

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
                elif node.startswith('a'):
                    data['kind'] = 'asset'
            print 'random: ', average_degree(G), len(G)
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
            print 'random: ', average_degree(G)
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
        elif type(add_fedfunds)==int:
            for node,data in G.nodes_iter(data=True):
                data['init_deg'] = G.degree(node)
            G.add_node('f01')
            banks = [node for node in G if node.startswith('b')]
            new_edges = [('f01',i) for i in np.random.choice(banks,add_fedfunds,replace=False,p=np.array(G.degree(banks).values())/sum(G.degree(banks).values()))]
            for i,j in new_edges:
                if j not in G:
                    print j
            G.add_edges_from(new_edges)
            G.node['f01']['kind']='fed'
        elif type(add_fedfunds) == str:
            cent = pd.Series(frag_centrality(G,1-1.0/av_deg)).sort_values()
            if add_fedfunds=='best':
                nodes = cent.tail(5000).index
            else:
                nodes = cent.head(5000).index
            G.add_node('f01')
            new_edges = [('f01',i) for i in nodes]
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
        if 'init_deg' in data.keys():
            break
        data['init_deg'] = G.degree(node)
    return G

fig = plt.figure()
gather_gc_by_avdeg = fig.gca()

fig2 = plt.figure()
gather_iters_by_avdeg = fig2.gca()
fail_counter = pd.DataFrame()
size_counter = pd.DataFrame()

for hit_probability in [0.0001]:
    for fract_threshold in [True]:
        num_of_banks = 10000
        num_of_assets = 10000
        list_to_plot=[]
        '''
        Threshold may be a fraction of the original or an absolute value
        '''
        th_base = np.arange(0.01,6,0.01)
        if fract_threshold:
            health_threshold = np.arange(0.5,1,0.001)
        else:
            health_threshold = np.arange(0,8,0.01)#, th_base/((1+th_base)**fract_threshold)#np.arange(0,1,0.01)#
#        hit_range = [0.001]#np.arange(0,.01,0.0001)
        av_deg_range = np.arange(10,40,5)#np.logspace(np.log10(0.01),np.log10(18),100)
        power = 2.5
        min_holdings = 5
        add_fedfunds = 'best'
        hit_probability = 0.001
        for kind in ['er']:#
            network_health = pd.DataFrame(columns = av_deg_range,index = health_threshold)
            iterations = network_health.copy()
            degree = network_health.copy()
            max_clust = network_health.copy()
            core_num = network_health.copy()
            bank_nodes = network_health.copy()
            asset_ndoes = network_health.copy()
            for avg in av_deg_range:
                round_rvs = generate_distribution(power,min_holdings,num_of_banks=num_of_banks,num_of_assets=num_of_assets,fit=False,fixed_mean=True,average=avg)
            #    fig = plt.figure()
            #    ax = fig.gca()
                single_net = False
                project_network = False
                link_temp = 0
            
                G = setup_network(num_of_banks=num_of_banks,num_of_assets=num_of_assets,round_rvs=round_rvs,p=power,kind=kind,min_holdings=min_holdings,\
                                  single_net=single_net,project_network=project_network,av_deg=avg,add_fedfunds=add_fedfunds)
                output = {}
#                link_counter = defaultdict(float)
                G_copy = G.copy()

                for th in health_threshold:
                    local_counter = []
                    local_size = []
    #                print th
                    G = G_copy.copy()
                    if project_network or single_net:
                        init_nodes_to_remove = ['b'+str(i) for i in random.sample(range(num_of_assets),int(hit_probability*num_of_assets))]
                    else:
                        init_nodes_to_remove = ['a'+str(i) for i in random.sample(range(num_of_assets),int(hit_probability*num_of_assets))]
                    link_temp = sum(G.degree(init_nodes_to_remove).values())
        #            link_counter[th]+=1.0/len(hit_range)*sum(G.degree(init_nodes_to_remove).values())/(sum(G.degree().values()))
                    G.remove_nodes_from(init_nodes_to_remove)
                    counter = 0
                    while True:
                        counter +=1
                        b_nodes_to_remove = set([node for node,data in G.nodes_iter(data=True) if \
                                         data['kind']=='bank' and not('f01' in G_copy[node].keys()) and (data['init_deg']==0 or G.degree(node)/(data['init_deg']**fract_threshold)<=th  or \
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
#                    link_counter[th]+=1.0/len(hit_range)*link_temp
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
                    output[th] = (counter,len(G),temp,temp2,temp3,temp4,temp5)
                    fail_counter = pd.concat([fail_counter,pd.Series(local_counter,name=(int(avg),th))],axis=1)
                    size_counter = pd.concat([size_counter,pd.Series(local_size,name=(int(avg),th))],axis=1)
                df = pd.DataFrame(output).T
                network_health[avg] = df[1]
                iterations[avg]=df[0]
                degree[avg] = df[2]
                max_clust[avg] = df[3]
                core_num[avg] = df[4]
                bank_nodes[avg] = df[5]
                asset_ndoes[avg] = df[6]
            if kind == 'powpow':
                if fract_threshold:
                    symbol = '.'
                else:
                    symbol = 'o'
            else:
                if fract_threshold:
                    symbol = '*'
                else:
                    symbol = '<'
#            fail_counter[fail_counter.count().nlargest(5).index].plot()
#            size_counter[size_counter.count().nlargest(5).index].plot()
            gather_gc_by_avdeg.plot(degree[degree.columns],max_clust[max_clust.columns],symbol,label='fractional' if fract_threshold else 'K-core')
            gather_gc_by_avdeg.set_xlabel('average degree')
            gather_gc_by_avdeg.set_ylabel('giant cluster')
            gather_gc_by_avdeg.legend(loc=4)
            gather_iters_by_avdeg.plot(degree[degree.columns],iterations[iterations.columns],symbol,label='fractional' if fract_threshold else 'K-core')
            gather_iters_by_avdeg.set_xlabel('average degree')
            gather_iters_by_avdeg.set_ylabel('iterations')
            gather_iters_by_avdeg.legend(loc=4)
            draw(health_threshold,av_deg_range,network_health,'Threshold','Average Degree','Surviving nodes',\
            'Surviving nodes vs Threshold and Average degree, Power = %.1f\nInit impact = %.3f, kind=%s, fract_th=%s'%(power,hit_probability,kind,fract_threshold))
        
#            draw(health_threshold,av_deg_range,iterations,'Threshold','Average Deree','Number of Iterations',\
#            'Number of Iterations vs Threshold and average degree\nPower = %.1f, Min Init Degree = %d, kind=%s'%(power,min_holdings,kind))
    #        
    #        draw(hit_range,np.arange(1.0,20),max_clust,'Impact Size','Threshold','Fraction of Giant component',\
    #        'Giant component fraction vs Threshold and Init impact\nPower = %.1f, Min Init Degree = %d, kind=%s'%(power,min_holdings,kind))
        
    #    draw(hit_range,health_threshold,core_num,'Impact Size','Threshold','Number of Max K-core',\
    #    'Maximal K-core vs Threshold and Init impact\nPower = %.1f, Min Init Degree = %d, kind=%s'%(power,min_holdings,kind))
    #    
    #    draw(hit_range,health_threshold,degree,'Impact Size','Threshold','Average degree',\
    #    'Average degree vs Threshold and Init impact\nPower = %.1f, Min Init Degree = %d, kind=%s'%(power,min_holdings,kind))
        
    #    draw(hit_range,health_threshold,bank_nodes,'Impact Size','Threshold','Surviving Banks',\
    #    'Banks vs Threshold and Init impact\nPower = %.1f, Min Init Degree = %d, kind=%s'%(power,min_holdings,kind))
    #    
    #    draw(hit_range,health_threshold,asset_ndoes,'Impact Size','Threshold','Surviving assets',\
    #    'Assets vs Threshold and Init impact\nPower = %.1f, Min Init Degree = %d, kind=%s'%(power,min_holdings,kind))
        #p0 = [0.5,-3]
        #fitfunc = lambda p,x: p[0]*x**p[1]
        #errfunc = lambda p,x,y: fitfunc(p,x) - y
        #p1,success = optimize.leastsq(errfunc,p0[:], args = (bins[:-1],hist))
        #ax3.set_title('Distribution of number of iteration\nfor Power Law network, decay = %.2f'%p1[1])
        #plt.show()
        #
#            network_health.index = health_threshold
#            axdf = network_health.T.plot(legend=False,title='Surviving nodes vs avg degree for various thresholds \npower=%.1f, Impact=%.3f, Kind=%s, fract th=%s'%(power,hit_probability,kind,fract_threshold))
#            axdf.set_xlabel('Average Degree')
#            axdf.set_ylabel('Surviving nodes')
    #    axdf = iterations.T.plot(legend=False,title='Number of iterations vs impact for various thresholds\npower=%.1f, MinDegree=%d, Kind=%s'%(power,min_holdings,kind))
    #    axdf.set_xlabel('Impact')
    #    axdf.set_ylabel('Iterations')
    #    axdf = max_clust.T.plot(legend=False,title='Max Cluster size vs impact for various thresholds\npower=%.1f, MinDegree=%d, Kind=%s'%(power,min_holdings,kind))
    #    axdf.set_xlabel('Impact')
    #    axdf.set_ylabel('Max Cluster fraction')
            '''
            most time spent on least intelligent bit :)
            useful line for visualisation of interesting bits:
            plt.plot(bpdf.diff().sort_values().values,'-o')
            when break point exhibits sharp transitions
            the number 25 in the break point calculation is a plug number for the dead network surviving nodes
            TODO consider doing the same for the .ix version of the plot
            '''
            break_point = [(col,min((network_health[col][network_health[col]<(0.01*(num_of_assets+num_of_banks))]).index)) for col in network_health.columns if len((network_health[col][network_health[col]<(0.01*(num_of_assets+num_of_banks))]).index)>0]
            bpdf = pd.Series(zip(*break_point)[1],index=zip(*break_point)[0])
            plt.figure()
            bpdf.plot()
    #    zu = bpdf.diff().sort_values()
    #    zu = zu[np.insert((np.diff(zu.values)>1e-6) | (np.diff(zu.values)==0),0,True)]
    #    if len(list_to_plot)==0:
    #        try:
    #            list_to_plot = sorted([b for a in zu.index[zu<min(zu[(zu.groupby(zu.values).count()>zu.groupby(zu.values).count().median()*2).values])] for b in network_health.index if np.isclose(a,b)])
    #        except:
    #            list_to_plot = []#[0,0.25,0.4,0.5,0.65]
    #    paper_plot(list_to_plot)#[0,0.25,0.28,0.33,0.375,0.4,0.425,0.44,0.5,0.6,0.665,0.71,0.75]
    #    paper_plot([0.1,0.22,0.25,0.29,0.31,0.36,0.65],True)
    
    #        ax.plot(zip(*break_point)[0],zip(*break_point)[1])#
    #        ax.set_xlabel('Threshold')
    #        ax.set_ylabel('Impact')
    #        ax.set_title('First Order phase transition, power=%.2f'%power)
    #        ax.legend(['Half Power Law','E-R'])
