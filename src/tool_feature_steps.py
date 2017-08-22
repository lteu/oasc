#! /usr/bin/env python

'''
learn_main <SCENARIO_NAME>


comments
========
coordinate macro learning process

variant of learn_optima_k.

Author: hearntest
'''

import os
import sys
import datetime
import time
import yaml
import csv
# from kit import *


#=======================================================
#=======================================================
#=======================================================


def getMiddleKEntries(dataset,mid=95.0):
  # get 95% middle dataset
  dataset.sort()
  start = round(len(dataset) * (1.0- mid/100.0))
  end = round(len(dataset) * (mid/100.0))
  
  dataset = dataset[int(start):int(end)]
  return dataset

def getAvgFeaturesCost(scenario_train):
  descriptionfile = scenario_train+'/'+'description.txt'
  cost_file = scenario_train +'/feature_costs.arff'
  status_file = scenario_train + '/feature_runstatus.arff'


  if not os.path.exists(cost_file):
    return None
  else:
    dic = loadDescriptionFile(descriptionfile)
    stepdic =  dic['feature_steps']
    feature_costs = {}
    step_cost_data = {}
    step_indexs = {}
    reader = csv.reader(open(cost_file), delimiter = ',')
    i = 2
    for row in reader:
      steps = set([])
      if row and '@ATTRIBUTE' in row[0].strip().upper()  \
      and 'instance_id' not in row[0] and 'repetition' not in row[0]:
        tmp_step = row[0].strip().split(' ')[1]
        
        if tmp_step in stepdic.keys():
          step_indexs[tmp_step] = i
          step_cost_data[tmp_step] = []
          i = i + 1
          
      elif row and row[0].strip().upper() == '@DATA':
        break

    # collect step data
    for row in reader:
      if len(row) < 2:
        continue
      # feature_costs[row[0]] = 0
      for (step,idx) in step_indexs.iteritems():
        # print step,idx
        if row[idx] != '?':
          step_cost_data[step].append(float(row[idx]))

    step_costs = {}
    for (step,values) in step_cost_data.iteritems():
      dataset = getMiddleKEntries(values)
      step_costs[step] = 2*round(sum(dataset)/len(dataset),2)
  
    return step_costs

def extendedSteps(stepdic,startstep):
  # print startstep
  if  'requires' not in stepdic[startstep]:
    return []
  else:
    return stepdic[startstep]['requires'] + extendedSteps(stepdic,stepdic[startstep]['requires'][0])

def necessaryfeatureSteps(scenario_train, selfeats):
  basic_steps = []
  required_steps = []
  descriptionfile = scenario_train+'/'+'description.txt'
  dic = loadDescriptionFile(descriptionfile)
  stepdic =  dic['feature_steps']

  for (key,val) in stepdic.iteritems():
    # get first level feature steps
    if set(val['provides']).intersection(selfeats): 
      basic_steps.append(key)
      if 'requires' in val and  val['requires'][0] not in required_steps:
        required_steps += val['requires']
  
  # get correlated feature steps
  path = []   
  for startstep in required_steps:
    path += extendedSteps(stepdic,startstep)

  # in enabler order
  path.reverse()
  full_steps = path + required_steps + basic_steps

  # remove duplicates
  # print sorted(set(full_steps), key=lambda x: full_steps.index(x))
  return sorted(set(full_steps), key=lambda x: full_steps.index(x))

def featureStepOrder(stepdic,selfeats):
  for (key,val) in stepdic.iteritems():
    # get first level feature steps
    if set(val['provides']).intersection(selfeats): 
      basic_steps.append(key)
      if 'requires' in val and  val['requires'][0] not in required_steps:
        required_steps += val['requires']
  
  # get correlated feature steps
  path = []   
  for startstep in required_steps:
    path += extendedSteps(stepdic,startstep)

  # in enabler order
  path.reverse()
  full_steps = path + required_steps + basic_steps
  return sorted(set(full_steps), key=lambda x: full_steps.index(x))

def loadDescriptionFile(descriptionfile):
  with open(descriptionfile) as stream:
    try:
        dic = yaml.load(stream)
        # print dic['number_of_feature_steps']
    except yaml.YAMLError as exc:
        print(exc)
    return dic


# testing

# def main(args):
#   # this program list all description information about feature step
#   root_arr = os.path.realpath(__file__).split('/')[:-2]
#   root = '/'.join(root_arr) 
#   src_path = root + '/src/'

#   train_path = root + '/data/oasc_scenarios/train/'

#   for subdir in os.listdir(train_path):
#     if not 'DS_Store' in  subdir:
#       descriptionfile = train_path+'/'+subdir+'/'+'description.txt'
#       scenario_train =  train_path+'/'+subdir
#       print subdir,
#       if not os.path.exists(descriptionfile):
#         sys.exit(descriptionfile+" not found")


#       dic = loadDescriptionFile(descriptionfile)
#       # print dic['number_of_feature_steps']
#       feature_steps =  dic['feature_steps']
#       print necessaryfeatureSteps(scenario_train, ['feature_100','feature_38','feature_51'])
#       print getAvgFeaturesCost(scenario_train)



# if __name__ == '__main__':
#   main(sys.argv[1:])
#   