'''
Helper module for computing the SUNNY schedule.
'''

import csv
import json
import sys
# from sunny_mock import get_schedule_fast
from math import sqrt
from combinations import binom, get_subset


def mine_solver(insts,inst_dic,portfolio_names,cutoff):
  if cutoff == 0 or len(insts) == 0:
    return [],len(insts)
  solved = { portfolio_names[i] : {"insts":[],"time":0.0,"count":0} for i in range(len(portfolio_names)) }
  for ins in insts:
    runs = inst_dic[ins]
    if isinstance(runs, str) :
      runs = eval(runs)
    for solver,infos in runs.items():
      if infos['info'] == 'ok':
        solved[solver]['insts'].append(ins)
        solved[solver]['time'] += infos['time']
        solved[solver]['count'] += 1
  cnt_isnt = 0
  cnt_time = 0
  best_solver = ''
  for solver,solverinfo in solved.items():
    if solverinfo['count'] > cnt_isnt:
      best_solver = solver
      cnt_isnt = solverinfo['count']
      cnt_time = solverinfo['time']
    elif solverinfo['count'] == cnt_isnt and solverinfo['time'] < cnt_time:
      best_solver = solver
      cnt_isnt = solverinfo['count']
      cnt_time = solverinfo['time']

  insts = set(insts) - set(solved[best_solver]['insts'])
  portfolio_names = [ pname for pname,pinfo in solved.items() if pinfo['count'] > 0]
  cutoff -= 1
  next_solver,unsolver = mine_solver(insts,inst_dic,portfolio_names,cutoff)
  return ([best_solver] + next_solver),unsolver


def promising_solver(inst_dic,portfolio_names,cutoff):

  solved_initally = { portfolio_names[i] : {"insts":[],"time":0.0,"count":0} for i in range(len(portfolio_names)) }
  insts = inst_dic.keys()
  for ins in insts:
    runs = inst_dic[ins]
    # print runs
    # sys.exit()
    if isinstance(runs, str) :
      runs = eval(runs)
    for solver,infos in runs.items():
      if infos['info'] == 'ok':
        solved_initally[solver]['insts'].append(ins)
        solved_initally[solver]['time'] += infos['time']
        solved_initally[solver]['count'] += 1

  best_pfolio,unsolver = mine_solver(insts,inst_dic,portfolio_names,cutoff)
  max_solved = len(insts) - unsolver
  return best_pfolio,solved_initally,max_solved

def get_schedule_fast(neighbours, timeout, portfolio, k, backup, schedule_size):
  cutoff = schedule_size
  inst_dic = neighbours
  portfolio_names = portfolio

  best_pfolio,solved_info,max_solved = promising_solver(inst_dic,portfolio_names,cutoff)

  n = sum([solved_info[s]['count'] for s in best_pfolio]) + (k - max_solved)
  schedule = {}
  # Compute the schedule and sort it by number of solved instances.
  for solver in best_pfolio:
    ns = solved_info[s]['count']
    if ns == 0 or round(timeout / n * ns) == 0:
      continue
    schedule[solver] = timeout / n * ns
  
  tot_time = sum(schedule.values())
  # Allocate to the backup solver the (eventual) remaining time.
  if round(tot_time) < timeout:
    if backup in schedule.keys():
      schedule[backup] += timeout - tot_time
    else:
      schedule[backup]  = timeout - tot_time
  sorted_schedule = sorted(schedule.items(), key = lambda x: solved_info[x[0]]['time'])
  return sorted_schedule

# def foldStats():
  
def normalize(feat_vector, selected_features, lims, inf, sup, def_feat_value):
  """
  Normalizes the feature vector in input in the range [inf, sup]
  """
  norm_vector = []
  for i in selected_features:
    lb = lims[str(i)][0] if str(i) in lims else lims[i][0]
    ub = lims[str(i)][1] if str(i) in lims else lims[i][1]
    f = feat_vector[i]
    if f == '?':
      f = def_feat_value
    else:
      f = float(f)
      if f < lb:
        f = inf
      elif f > ub:
        f = sup
      else:
        if ub - lb != 0:
          x = (f - lb)/(ub - lb)
          f = inf + (sup - inf) * x
        else:
          f = def_feat_value
        assert inf <= f <= sup
    norm_vector.append(f)
  return norm_vector

def sel_inst_by_rate(cleandist):
  rlt = []
  pre_val = 0
  pre_diff = 9999
  for x in cleandist:
    if x - pre_val == 0:
      rlt.append(x)
      continue

    new_diff = x - pre_val
    percent = float(new_diff)/float(pre_diff)
    if percent > 12:
      break

    rlt.append(x)
    pre_val = x
    pre_diff = new_diff

  return len(rlt)

def get_neighbours_r(feat_vector, selected_features, kb, k,r):
  """
  Returns a dictionary (inst_name, inst_info) of the k instances closer to the 
  feat_vector in the knowledge base kb.
  """
  reader = csv.reader(open(kb, 'r'), delimiter = '|')
  infos = {}
  distances = []
  for row in reader:
    inst = row[0]
    r_i = r[inst]
    if isinstance(row[1], str):
      ori_vector = json.loads(row[1])
      ori_vector = [float(i) for i in ori_vector]
    else:
      ori_vector = row[1]

    d = euclidean_distance(
      feat_vector, [ori_vector[i] for i in selected_features]
    )/r_i
    distances.append((d, inst))
    infos[inst] = row[2]
  distances.sort(key = lambda x : x[0])

  return dict((inst, infos[inst]) for (d, inst) in distances[0 : k])



def get_neighbours(feat_vector, selected_features, kb, k):
  """
  Returns a dictionary (inst_name, inst_info) of the k instances closer to the 
  feat_vector in the knowledge base kb.
  """
  reader = csv.reader(open(kb, 'r'), delimiter = '|')
  return get_neighbours_list(feat_vector, selected_features, reader, k)


def get_neighbours_list(feat_vector, selected_features, array, k):
  infos = {}
  distances = []
  for row in array:
    inst = row[0]

    if isinstance(row[1], str):
      ori_vector = json.loads(row[1])
      ori_vector = [float(i) for i in ori_vector]
    else:
      ori_vector = row[1]

    d = euclidean_distance(
      feat_vector, [ori_vector[i] for i in selected_features]
    )
    distances.append((d, inst))
    infos[inst] = row[2]
  distances.sort(key = lambda x : x[0])

  return dict((inst, infos[inst]) for (d, inst) in distances[0 : k])

def euclidean_distance(fv1, fv2):
  """
  Computes the Euclidean distance between two feature vectors fv1 and fv2.
  """
  assert len(fv1) == len(fv2)
  distance = 0.0
  for i in range(0, len(fv1)):
    d = fv1[i] - fv2[i]
    distance += d * d
  return sqrt(distance)


def get_schedule_proposed(neighbours, timeout, portfolio, k, backup, max_size, schedule_size,sharemode):
  """
  Returns the corresponding SUNNY schedule.
  Tong: max_size and schedule_size are ambiguous, in pratical,
  max_size = len(porfolio), 
  schedule_size  = limit of proposed schedule, 5 as default
  """

  # Dictionaries for keeping track of the instances solved and the runtimes. 
  solved = {} # solver's successful instances
  times  = {} # solver's total time
  for solver in portfolio:
    solved[solver] = set([])
    times[solver]  = 0.0

  for inst, item in neighbours.items():
    if isinstance(item, str):
      item = eval(item)
    for solver in portfolio:
      time = item[solver]['time']
      if time < timeout:
        solved[solver].add(inst)
      times[solver] += time
  # Select the best sub-portfolio, i.e., the one that allows to solve more 
  # instances in the neighborhood.
  max_solved = 0
  min_time = float('+inf')
  best_pfolio = []
  m = schedule_size
  for i in range(1, m + 1):
    old_pfolio = best_pfolio
    
    for j in range(0, binom(max_size, i)):
      solved_instances = set([])
      solving_time = 0
      # get the (j + 1)-th subset of cardinality i
      sub_pfolio = get_subset(j, i, portfolio)
      for solver in sub_pfolio:
        solved_instances.update(solved[solver])
        solving_time += times[solver]
      num_solved = len(solved_instances)
      
      if num_solved >  max_solved or \
        (num_solved == max_solved and solving_time < min_time):
          min_time = solving_time
          max_solved = num_solved
          best_pfolio = sub_pfolio
          
    if old_pfolio == best_pfolio:
      break
    
  # old slot assignment approach
  # n is the number of instances solved by each solver plus the instances 
  # that no solver can solver.
  n = sum([len(solved[s]) for s in best_pfolio]) + (k - max_solved)
  schedule = {}
  # Compute the schedule and sort it by number of solved instances.
  for solver in best_pfolio:
    ns = len(solved[solver])
    if ns == 0 or round(timeout / n * ns) == 0:
      continue
    schedule[solver] = timeout / n * ns


  # new slot assignment approach
  # schedule = {}
  # for solver in best_pfolio: 
  #   suggested_time = get_solver_time_slot(solver,neighbours,timeout)
  #   schedule[solver] = suggested_time


  # Allocate to the backup solver the (eventual) remaining time.
  tot_time = sum(schedule.values())
  selected_solvers = schedule.keys()

  time_remain = timeout - round(tot_time)

  if time_remain > 0:
    # print time_remain,' ',schedule
    # way of sharing: equal share, propotional share, all backup
    
    # equale share
    if sharemode == 'equal':
      slot_remain = time_remain/(len(selected_solvers)+1)
      for aSolver in selected_solvers:
        schedule[aSolver] += slot_remain
      if backup in schedule.keys():
        schedule[backup] += slot_remain
      else:
        schedule[backup] = slot_remain

    # all backup
    elif sharemode == 'allbackup':
      if backup in schedule.keys():
        schedule[backup] += timeout - tot_time
      else:
        schedule[backup]  = timeout - tot_time

    # propotional share
    elif sharemode == 'propotional':
      allSolved = sum([len(solved[s]) for s in best_pfolio])
      for aSolver in selected_solvers:
        schedule[aSolver] += time_remain / allSolved * len(solved[aSolver])

  # print schedule
  sorted_schedule = sorted(schedule.items(), key = lambda x: times[x[0]])
  return sorted_schedule

# Tong: several tests demonstrate that it is not good
# def get_solver_time_slot(solver,neighbours,timeout):
#   suggested_time = -1
#   for inst, item in neighbours.items():
#     item = eval(item)
#     time = item[solver]['time']
#     if time < timeout and time > suggested_time:
#       suggested_time = time * 3/2
#   return suggested_time


def get_schedule_maxi(neighbours, timeout, portfolio, k, backup, max_size):
  inst_dic = neighbours
  portfolio_names = portfolio

  solved_initally = { portfolio_names[i] : {"insts":[],"time":0.0,"count":0} for i in range(len(portfolio_names)) }
  insts = inst_dic.keys()
  for ins in insts:
    runs = inst_dic[ins]
    if isinstance(runs, str):
      runs = eval(inst_dic[ins])
    for solver,infos in runs.items():
      if infos['info'] == 'ok':
        solved_initally[solver]['insts'].append(ins)
        solved_initally[solver]['time'] += infos['time']
        solved_initally[solver]['count'] += 1

  performances = [(k,v['time']) for k,v in solved_initally.iteritems()]  
  performances = sorted(performances, key=lambda x: x[1],reverse=True)

  return [[performances[0][0],99999]]


def get_schedule(neighbours, timeout, portfolio, k, backup, max_size):
  """
  Returns the corresponding SUNNY schedule.
  """

  # Dictionaries for keeping track of the instances solved and the runtimes. 
  solved = {} # solver's successful instances
  times  = {} # solver's total time
  for solver in portfolio:
    solved[solver] = set([])
    times[solver]  = 0.0
  for inst, item in neighbours.items():
    if isinstance(item, str):
      item = eval(item)
    for solver in portfolio:
      time = item[solver]['time']
      if time < timeout:
        solved[solver].add(inst)
      times[solver] += time

  # Select the best sub-portfolio, i.e., the one that allows to solve more 
  # instances in the neighborhood.
  max_solved = 0
  min_time = float('+inf')
  best_pfolio = []
  m = max_size
  for i in range(1, m + 1):
    old_pfolio = best_pfolio
    
    for j in range(0, binom(m, i)):
      solved_instances = set([])
      solving_time = 0
      # get the (j + 1)-th subset of cardinality i
      sub_pfolio = get_subset(j, i, portfolio)
      for solver in sub_pfolio:
        solved_instances.update(solved[solver])
        solving_time += times[solver]
      num_solved = len(solved_instances)
      
      if num_solved >  max_solved or \
        (num_solved == max_solved and solving_time < min_time):
          min_time = solving_time
          max_solved = num_solved
          best_pfolio = sub_pfolio
          
    if old_pfolio == best_pfolio:
      break

  # n is the number of instances solved by each solver plus the instances 
  # that no solver can solver.
  n = sum([len(solved[s]) for s in best_pfolio]) + (k - max_solved)
  schedule = {}
  # Compute the schedule and sort it by number of solved instances.
  for solver in best_pfolio:
    ns = len(solved[solver])
    if ns == 0 or round(timeout / n * ns) == 0:
      continue
    # print solver,n,ns
    schedule[solver] = timeout / n * ns
  
  # tot_time = sum(schedule.values())
  tot_time = round(sum(schedule.values()),5)
  # Allocate to the backup solver the (eventual) remaining time.
  if tot_time < timeout:
    if backup in schedule.keys():
      schedule[backup] += timeout - tot_time
    else:
      schedule[backup]  = timeout - tot_time
  sorted_schedule = sorted(schedule.items(), key = lambda x: times[x[0]])
  #assert sum(t for (s, t) in sorted_schedule) - timeout < 0.001


  return sorted_schedule

def get_sunny_schedule(
  lb, ub, def_feat_value, kb_path, kb_name, static_schedule, timeout, k, \
  portfolio, backup, selected_features, feat_vector, total_cost, max_size,maximize
):
  selected_features = sorted(selected_features.values())
  with open(kb_path + kb_name + '.lims') as infile:
    lims = json.load(infile)
  
  norm_vector = normalize(
    feat_vector, selected_features, lims, lb, ub, def_feat_value
  )
  kb = kb_path + kb_name + '.info'
  neighbours = get_neighbours(norm_vector, selected_features, kb, k)

  timeout -= total_cost + sum(t for (s, t) in static_schedule)
  if timeout > 0: 
    schedule = get_schedule(neighbours, timeout, portfolio, k, backup, max_size) if not maximize else get_schedule_maxi(neighbours, timeout, portfolio, k, backup, max_size)
    return schedule
  else:
    return []

def get_sunny_schedule_r(
  lb, ub, def_feat_value, kb_path, kb_name, static_schedule, timeout, k, \
  portfolio, backup, selected_features, feat_vector, feat_cost, max_size,r
):
  selected_features = sorted(selected_features.values())
  with open(kb_path + kb_name + '.lims') as infile:
    lims = json.load(infile)
  
  norm_vector = normalize(
    feat_vector, selected_features, lims, lb, ub, def_feat_value
  )
  kb = kb_path + kb_name + '.info'
  neighbours = get_neighbours_r(norm_vector, selected_features, kb, k,r)
  timeout -= feat_cost + sum(t for (s, t) in static_schedule)
  if timeout > 0: 
    schedule = get_schedule(neighbours, timeout, portfolio, k, backup, max_size)
    # schedule = get_schedule_fast(neighbours, timeout, portfolio, k, backup, 2)

    return schedule

    # return get_schedule_proposed(neighbours, timeout, portfolio, k, backup, max_size,5,'equal') # sharemode:equal, allbackup,propotional
  else:
    return []