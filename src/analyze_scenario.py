#! /usr/bin/env python

'''
Functions for the statistics

analyze_scenario_stats for conventional aslib statistics

analyze_log for number of solvers

metrics aligned with the benchmark rules:
https://docs.google.com/spreadsheets/d/1HqzXcsFpQHrTWiu_BLxxajFCBbGTj8YgsB6Lwk7xgOo/pubhtml?gid=0&single=true

'''
from __future__ import division
import os
import sys
import json
import getopt

from sunny import *

def getRunTime(path_scenario):
  # print 'Extracting runtimes'
  
  reader = csv.reader(open(path_scenario + '/algorithm_runs.arff'), delimiter = ',')
  runtimes = {}
  for row in reader:
    if row and row[0].strip().upper() == '@DATA':
      # Iterates until preamble ends.
      break
  for row in reader:
    inst = row[0]
    solv = row[2]
    time = float(row[3])
    info = row[4]
    if inst not in runtimes.keys():
      runtimes[inst] = {}
    runtimes[inst][solv] = [info, time]

  runtimes = remove_unsolved(runtimes)
  return runtimes

def remove_unsolved(runtimes):
  newRunTime = {}
  for inst, solvers in runtimes.iteritems():
    solved = False
    for solver, aTuple in solvers.iteritems():
      if aTuple[0] == 'ok':
        solved = True
    if solved == True:
      newRunTime[inst] = solvers
    # runtimes[inst]['state'] = solved


  # runtimes = dict((k,v) for k,v in runtimes.iteritems() if v['state'] == True)
  return newRunTime

def analyze_scenario_stats(path_scenario):
  runtimes = getRunTime(path_scenario)
  # print len(runtimes)
  # print runtimes
  # runtimes = remove_unsolved(runtimes)
  # print len(runtimes)
  # print runtimes

  # statistics and evaluation 
  fsi = 0.0
  fsi_vbs = 0.0
  fsi_sbs = 0.0
  par10 = 0.0
  par10_vbs = 0.0
  par10_sbs = 0.0
  n = 0
  m = 0
  p = 0
  
  scenario = path_scenario.split('/')[-1]


  path = path_scenario
  dirpath = path + '/cv_' + scenario
  for subdir in os.listdir(dirpath):
    test_dir = subdir.replace('train_', 'test_')
    pred_file = dirpath +'/'+test_dir+'/predictions.csv'
    kb_name = subdir.split('/')[-1]
    
    if 'train_' in subdir and 'kb_' not in subdir:

      if not os.path.exists(pred_file):
        print 'Not computed ...',pred_file,
        continue

      if os.path.exists(path + '/feature_costs.arff'):
        # print 'Extracting feature costs'
        args_file =dirpath+'/'+ subdir + '/kb_' + kb_name + '/kb_' + kb_name + '.args'
        with open(args_file) as infile:
          args = json.load(infile)

        feature_steps = args['feature_steps']
        feature_cost = {}
        reader = csv.reader(open(path + '/feature_costs.arff'), delimiter = ',')
        for row in reader:
          steps = set([])
          i = 2
          if row and '@ATTRIBUTE' in row[0] \
          and 'instance_id' not in row[0] and 'repetition' not in row[0]:
            if row[0].strip().split(' ')[1] in feature_steps:
              steps.add(i)
            i += 1
          elif row and row[0].strip().upper() == '@DATA':
            # Iterates until preamble ends.
            break
        for row in reader:
          feature_cost[row[0]] = 0
          for i in steps:
            if row[i] != '?':
              feature_cost[row[0]] += float(row[i])
      
      # print 'Computing fold statistics'

      # print dirpath,subdir,pred_file
      reader = csv.reader(open(pred_file), delimiter = ',')


      old_inst = ''
      par = True
      args_file = dirpath +'/'+subdir+ '/kb_' + kb_name + '/kb_' + kb_name + '.args'
      kb_args = {}
      # print args_file
      with open(args_file) as infile:
        kb_args = json.load(infile)
        timeout = kb_args['timeout']

      # sbs
      sbs = kb_args['backup']
      instances = []

      first = True
      for row in reader:
        if first:
          first = False
          continue
        inst = row[0]
        if inst not in runtimes:
          continue
        instances.append(inst)
        if inst == old_inst:
          if par:
            continue
        else:
          if not par:
            par10 += timeout * 10
            par = True
            p += 1
          n += 1
          if os.path.exists(path + '/feature_costs.arff'):
            time = feature_cost[inst]
          else:
            time = 0.0
          times = [x[1] for x in runtimes[inst].values() if x[0] == 'ok']
          if times:
            m += 1
            fsi_vbs += 1
            par10_vbs += min(times)
          else:
            par10_vbs += 10 * timeout

        old_inst = inst
        solver = row[2]
        solver_time = float(row[3])
        if  runtimes[inst][solver][0] == 'ok' \
        and runtimes[inst][solver][1] <= solver_time:
          par = True
          if time + runtimes[inst][solver][1] >= timeout:
            par10 += 10 * timeout
            p += 1
          else:
            fsi += 1
            par10 += time + runtimes[inst][solver][1]
        elif time + min([solver_time, runtimes[inst][solver][1]]) < timeout:
          time += min([solver_time, runtimes[inst][solver][1]])
          par = False
        else:
          par10 += 10 * timeout
          par = True
          p += 1
    
      if not par:
        par10 += timeout * 10
        par = True
        p += 1

      # tong: sbs
      instances = list(set(instances))
      for inst in instances:
        tmp_par10 = 0
        if  runtimes[inst][sbs][0] == 'ok' \
        and runtimes[inst][sbs][1] <= timeout:
          fsi_sbs += 1  
          par10_sbs += runtimes[inst][sbs][1]
          tmp_par10 = runtimes[inst][sbs][1]
        else:
          par10_sbs += 10 * timeout
          tmp_par10 = 10 * timeout
        # print inst,tmp_par10
        
  assert p + fsi == n


  return  fsi/n, fsi_vbs/n, par10/n, par10_vbs/n,m,n,scenario, fsi_sbs/n, par10_sbs/n


def analyze_log(path_scenario):

  scenario = path_scenario.split('/')[-1]
  path = path_scenario
  dirpath = path + '/cv_' + scenario
  count = 0
  solvern = 0
  for subdir in os.listdir(dirpath):
    kb_name = subdir.split('/')[-1]
    if 'train_' in subdir and 'kb_' not in subdir:
      log_file =dirpath+'/'+ subdir + '/kb_' + kb_name + '/execution_log.json'
      with open(log_file) as infile:
        log = json.load(infile)
        # print  log['number_solver'],log_file
        solvern += log['number_solver']
        count += 1

  k_log = log['k']
  print 'solvers used : ',round(float(solvern/count),2), 'k', k_log


# print sys.argv[1]

