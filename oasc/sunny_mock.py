'''
Helper module for computing the SUNNY schedule with

mocked SUNNY functions

Author: Heartnest
'''

import copy
import csv
import sys
import os
import getopt
from math import isnan, sqrt
from math import sqrt

root_arr = os.path.realpath(__file__).split('/')[:-2]
root = '/'.join(root_arr) 
src_path = root + '/src/'
sys.path.append(src_path)

from train_scenario_oasc import parse_description
from train_scenario_oasc import parse_arguments as parse_train_arguments
from sunny_oasc import normalize,get_neighbours_list
from pre_process import select_manually_str

def test_scenario_mock(input_args,test_path,args,lims,info,test,ground):
  '''
  main function
  '''
  k, lb, ub, feat_def,   static_schedule, timeout,      \
  portfolio, backup,  selected_features,  feature_steps = parse_test_arg_mock(args)
  
  test_insts = test[0]
  feature_costs = test[1]

  solution = []
  for instt in test_insts:
    inst = instt[0]
    feat_vector = instt[1]
    if feature_costs:
      feat_cost = feature_costs[inst]
    else:
      feat_cost = 0
    max_size = len(portfolio)

    # Get the schedule computed by SUNNY algorithm.
    schedule = get_sunny_schedule_mock(
      lb, ub, feat_def, lims, info, static_schedule, timeout, k, \
      portfolio, backup, selected_features, feat_vector, feat_cost, max_size
    )
    
    i = 1
    for (s, t) in schedule:
      row = inst + ',' + str(i) + ',' + s + ',' + str(t)
      solution.append(row.split(','))
      i += 1


  par10 = solution_score_mock(args,solution,feature_costs,ground)

  return par10

  

# train
#=============================

def pre_process_mock(input_args,args_in):
  '''
  prepocess for feature seletion
  '''
  args = copy.copy(args_in)
  # args = args_in
  scenario, evaluator, search, filter_portfolio = parse_pre_arguments_mock(input_args)

  # Feature selection.
  if evaluator == 'wrapper':
    sel_feat_strs = search.split(',')
    selected_features, feature_steps = select_manually_str(args,sel_feat_strs)
    args['selected_features'] = selected_features
    args['feature_steps'] = feature_steps
  elif evaluator and search:
    # to-do
    selected_features, feature_steps = select_features(
      args, info_file, scenario, evaluator, search, filter_portfolio
    )
    args['selected_features'] = selected_features
    args['feature_steps'] = feature_steps
  
  return args

def train_scenario_mock(input_args):
  '''
  train scenario for args, info, lims
  comment: due to file access, this should be called a priori
  '''
  scenario, lb, ub, feat_def, feat_timeout, discard, kb_path, kb_name, root_path, value_k, discardAll = \
    parse_train_arguments(input_args)

  description_path = scenario
  if root_path != '':
    description_path = root_path

  pfolio, timeout, num_features, feature_steps = parse_description(description_path)
  
  if feat_timeout < 0:
    feat_timeout = timeout / 2

  # Processing runtimes.
  reader = csv.reader(open(scenario + 'algorithm_runs.arff'), delimiter = ',')
  for row in reader:
    if row and row[0].strip().upper() == '@DATA':
      # Iterates until preamble ends.
      break
  longest_time = 0
  kb = {}
  solved = dict((s, [0, 0.0]) for s in pfolio)
  for row in reader:
    if len(row) < 2:
      continue
    inst   = row[0]
    solver = row[2]
    info   = row[4]
    if info != 'ok':
      time = timeout
    else:
      time = float(row[3])
      solved[solver][0] += 1

    if time > longest_time:
      longest_time = time

    solved[solver][1] += time
    
    if inst not in kb.keys():
      kb[inst] = {}
    kb[inst][solver] = {'info': info, 'time': time}
  # Backup solver.
  backup = min((-solved[s][0], solved[s][1], s) for s in solved.keys())[2]

  # Processing features.
  cost_file = scenario + 'feature_costs.arff'
  if os.path.exists(cost_file):
    reader = csv.reader(open(scenario + 'feature_costs.arff'), delimiter = ',')
    i = 0
    # fn[i] is the name of the i-th feature step.
    fn = {}
    for row in reader:
      if row and '@ATTRIBUTE' in row[0].strip().upper()  \
      and 'instance_id' not in row[0] and 'repetition' not in row[0]:
        fn[i] = row[0].strip().split(' ')[1]
        i += 1
      elif row and row[0].strip().upper() == '@DATA':
        # Iterates until preamble ends.
        break
    for row in reader:
      i = 0
      for cost in row[2:]:
        if cost != '?' and float(cost) > feat_timeout and \
        fn[i] in feature_steps.keys():
          del feature_steps[fn[i]]
        i += 1
    selected_features = []
    for fs in feature_steps.values():
      selected_features += fs

 
  reader = csv.reader(open(scenario + 'feature_values.arff'), delimiter = ',')
  i = 0
  # fn[i] is now the name of the i-th feature.
  fn = {}
  for row in reader:
    if row and '@ATTRIBUTE' in row[0].strip().upper()  \
    and 'instance_id' not in row[0] and 'repetition' not in row[0]:
      fn[i] = row[0].strip().split(' ')[1]
      i += 1
    elif row and row[0].strip().upper() == '@DATA':
      # Iterates until preamble ends.
      break
  features = {}
  lims = {}
  instances = set([])
  for row in reader:
    if len(row) < 2:
      continue
    inst = row[0]
    if inst not in instances:
      instances.add(inst)
    nan = float("nan")
    feat_vector = []
    for f in row[2:]:
      if f == '?':
        feat_vector.append(float("nan"))
      else:
        feat_vector.append(float(f))
    if not lims:
      for k in range(0, len(feat_vector)):
        lims[k] = [float('+inf'), float('-inf')]
    # Computing min/max value for each feature.
    for k in range(0, len(feat_vector)):
      if not isnan(feat_vector[k]):
        if feat_vector[k] < lims[k][0]:
          lims[k][0] = feat_vector[k]
        if feat_vector[k] > lims[k][1]:
          lims[k][1] = feat_vector[k]
    features[inst] = feat_vector
    # print len(feat_vector),num_features
    assert len(feat_vector) == num_features

  # Scaling features.
  info = []
  test_count = 0
  for (inst, feat_vector) in features.items():

    if discard and not [s for s, it in kb[inst].items() if it['info'] == 'ok']:
      # Instance not solvable by any solver.
      continue

    if discardAll and len([s for s, it in kb[inst].items() if it['info'] == 'ok']) == len(kb[inst]):
      # Instance solvable by all solvers.
      continue

    test_count += 1
    new_feat_vector = []
    for k in range(0, len(feat_vector)):
      # Constant or not numeric feature.
      if lims[k][0] == lims[k][1] or isnan(feat_vector[k]):
        new_val = feat_def
      else:
        min_val = lims[k][0]
        max_val = lims[k][1]
        # Scale feature value in [lb, ub].
        x = (feat_vector[k] - min_val) / (max_val - min_val)
        new_val = lb + (ub - lb) * x
      assert lb <= new_val <= ub
      new_feat_vector.append(new_val)
    assert nan not in new_feat_vector
    info.append([inst, new_feat_vector, kb[inst]])
  
  # Creating <KB>.args
  neigh_size = int(round(sqrt(len(instances)))) if value_k == -1 else value_k

  if int(timeout) == 0: #incase, missing timeout information
    timeout = 2 * longest_time

  args = {
    'lb': lb,
    'ub': ub,
    'feat_def': feat_def,
    'backup': backup,
    'timeout': timeout,
    'portfolio': pfolio,
    'neigh_size': neigh_size,
    'static_schedule': [],
    'selected_features': dict(
      (fn[k], k)
      for k, v in lims.items() 
      if v[0] != v[1] and 
      (not os.path.exists(cost_file) or fn[k] in selected_features)
    ), 
    'feature_steps': feature_steps,
  }

  return args,info,lims



# SUNNY
#=============================

def get_sunny_schedule_mock(
  lb, ub, def_feat_value, lims, info, static_schedule, timeout, k, \
  portfolio, backup, selected_features, feat_vector, feat_cost, max_size
):

  selected_features = sorted(selected_features.values())

  # normalize test instance by value and selected features
  norm_vector = normalize(
    feat_vector, selected_features, lims, lb, ub, def_feat_value
  )

  neighbours =  get_neighbours_list(norm_vector, selected_features, info, k)

  timeout -= feat_cost + sum(t for (s, t) in static_schedule)

  if timeout > 0: 
    # to improve!

    return get_schedule_fast(neighbours, timeout, portfolio, k, backup, 3)
    # return get_schedule(neighbours, timeout, portfolio, k, backup, max_size) # sharemode:equal, allbackup,propotional
  else:

    return []


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



def promising_solver(inst_dic,portfolio_names,cutoff):
  '''
  calculate schedule
  '''
  solved_initally = { portfolio_names[i] : {"insts":[],"time":0.0,"count":0} for i in range(len(portfolio_names)) }
  insts = inst_dic.keys()
  for ins in insts:
    runs = inst_dic[ins]
    for solver,infos in runs.items():
      if infos['info'] == 'ok':
        solved_initally[solver]['insts'].append(ins)
        solved_initally[solver]['time'] += infos['time']
        solved_initally[solver]['count'] += 1

  best_pfolio,unsolver = mine_solver(insts,inst_dic,portfolio_names,cutoff)
  max_solved = len(insts) - unsolver
  return best_pfolio,solved_initally,max_solved


def mine_solver(insts,inst_dic,portfolio_names,cutoff):
  '''
  recursive solver to find schedule
  '''
  if cutoff == 0 or len(insts) == 0:
    return [],len(insts)
  solved = { portfolio_names[i] : {"insts":[],"time":0.0,"count":0} for i in range(len(portfolio_names)) }
  for ins in insts:
    runs = inst_dic[ins]
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


# utility function
#=============================

def solution_score_mock(args,solution,feature_cost,runtimes):
  '''
  calculate solution score
  '''
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

  timeout = args['timeout']
  sbs = args['backup']
  instances = []
  old_inst = ''
  par = True
  first = True
  for row in solution:
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
      if feature_cost:
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

  assert p + fsi == n

  fsi_score = fsi/n
  # print 'FSI SCORE', fsi_score
  return  par10/n

# supports
#=============================

def parse_pre_arguments_mock(input_args):
  '''
  Parse the options specified by the user and returns the corresponding
  arguments properly set.
  '''
  try:
    opts, param_args = getopt.getopt(
      input_args, 'S:E:', ['help', 'static-schedule', 'filter-portfolio', 'kb-path=']
    )
  except getopt.GetoptError as msg:
    print >> sys.stderr, msg
    sys.exit(2)

  scenario = param_args[0]
  if scenario[-1] != '/':
    scenario += '/'
    
  # Initialize variables with default values.
  feat_algorithm = None
  evaluator = ''
  search = ''
  filter_portfolio = False

  # Options parsing.
  for o, a in opts:
    if o == '-E':
      evaluator = a
    elif o == '-S':
      search = a
    elif o == '--filter-portfolio':
      filter_portfolio = True

  return scenario, evaluator, search, filter_portfolio



def parse_test_arg_mock(args):
  lb = args['lb']
  ub = args['ub']
  k = args['neigh_size']
  backup = args['backup']
  timeout = args['timeout']
  feat_def = args['feat_def']
  portfolio = args['portfolio']
  feature_steps = args['feature_steps']
  static_schedule = args['static_schedule']
  selected_features = args['selected_features']
  return k, lb, ub, feat_def,   static_schedule, timeout,      \
  portfolio, backup,  selected_features,  feature_steps


def getFeatureCost(scenario,feature_steps):
  cost_file = scenario + 'feature_costs.arff'
  feature_costs = {}

  if os.path.exists(cost_file):
    reader = csv.reader(open(cost_file), delimiter = ',')
    for row in reader:
      steps = set([])
      i = 2
      if row and '@ATTRIBUTE' in row[0].strip().upper()  \
      and 'instance_id' not in row[0] and 'repetition' not in row[0]:
        if row[0].strip().split(' ')[1] in feature_steps.keys():
          steps.add(i)
        i += 1
      elif row and row[0].strip().upper() == '@DATA':
        # Iterates until preamble ends.
        break
    for row in reader:
      feature_costs[row[0]] = 0
      for i in steps:
        if row[i] != '?':
          feature_costs[row[0]] += float(row[i])
  return feature_costs

