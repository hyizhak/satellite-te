# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:44:17 2024

@author: admin
"""

import pickle
# =========== Import Iridium Topologies ========
file = open('IridiumDataSet14day20sec/Iridium_DataSetForAgent_Day0.pkl','rb')
data = pickle.load(file)
file.close()
FlowSet = data[0]
Adj_Matrix = data[1]
ISLCap = data[2]
UpLinkCap = data[3]
DownLinkCap = data[4]