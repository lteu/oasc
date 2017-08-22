#! /usr/bin/env python

'''
train_scenario [OPTIONS] <SCENARIO_PATH>

Creates a SUNNY knowledge base corresponding to the ASlib scenario contained in 
<SCENARIO_PATH>. A knowledge base <KB> is basically a folder containing 3 files:

  <KB>.infos  csv file containing the runtimes and the feature vectors 
  
  <KB>.lims   python dict containing the lower/upper bounds for each feature
  
  <KB>.args   python dict containing the arguments needed by SUNNY algorithm


Options
=======
  --feat-range <LB>,<UB>
   Scales the feature in the range [LB, UB]. The default value is -1,1.
  --feat-def <VALUE>
   The default value for a missing/non-numeric feature, by default set to -1.
  --feat-timeout <TIME>
   Sets the timeout for a feature step: if <TIME> is exceeded, all the features 
   belonging to such step are removed. By default is set to T/2, where T is the 
   solving timeout for the scenario.
  --discard
   Discards all the instances not solvable by any solver. Unset by default.
  --discardAll
   Discards all the instances not solvable by any solver AND solvable by all solvers. Unset by default.
  --kb-path <PATH>
   Creates the SUNNY knowledge base at the specified path. By default, it is set
   to <SCENARIO_PATH>
  --kb-name <NAME>
   Creates the SUNNY knowledge base with the specified name, i.e., it creates a 
   folder <PATH>/<NAME> containing the knowledge base (where <PATH> is the given 
   knowledge base path). The default name is kb_<SCENARIO>, where <SCENARIO> is 
   the name of the folder containing the given scenario.
  --help
   Prints this message.

Newly introduced key in args:

feature_base: for backup
feature_wrapper: for storing selected features by wrapper

comments
========
This script will consider: value bounds of features,
'''

import os
import csv
import sys
import json
import getopt
import yaml
from math import isnan, sqrt
from tool_feature_steps import getAvgFeaturesCost

def parse_arguments(args):
  '''
  Parse the options specified by the user and returns the corresponding
  arguments properly set.
  '''
  try:
    options = [
      'feat-range=', 'feat-def=', 'feat-timeout=', 'kb-path=', 'kb-name=',
      'help', 'discard','discardAll','root-path=','value-k='
    ]
    opts, args = getopt.getopt(args, None, options)
  except getopt.GetoptError as msg:
    print >> sys.stderr, msg
    print >> sys.stderr, 'For help use --help'
    sys.exit(2)
  
  if not args:
    if not opts:
      print >> sys.stderr, 'Error! No arguments given.'
      print >> sys.stderr, 'For help use --help'
      sys.exit(2)
    else:
      print __doc__
      sys.exit(0)

  scenario = args[0]
  if scenario[-1] != '/':
    scenario += '/'
  if not os.path.exists(scenario):
    print >> sys.stderr, 'Error: Directory ' + scenario + ' does not exists.'
    print >> sys.stderr, 'For help use --help'
    sys.exit(2)
    
  # Initialize the variables with default values.
  lb = -1
  ub =  1
  feat_def = -1
  feat_timeout = -1
  discard = False
  discardAll = False
  kb_path = scenario
  kb_name = 'kb_' + scenario.split('/')[-2]
  root_path = ''
  value_k = -1
  # Options parsing.
  for o, a in opts:
    if o == '--help':
      print __doc__
      sys.exit(0)
    elif o == '--feat-range':
      s = a.split(',')
      lb = float(s[0])
      ub = float(s[1])
    elif o == '--feat-def':
      feat_def = float(a)
    elif o == '--feat-timeout':
      feat_timeout = float(a)
    elif o == '--discard':
      discard = True
    elif o == '--discardAll':
      discardAll = True
    elif o == '--kb-path':
      if not os.path.exists(a):
        print >> sys.stderr, 'Error! Directory ' + a + ' not exists.'
        print >> sys.stderr, 'For help use --help'
        sys.exit(2)
      if a[-1] == '/':
        kb_path = a[:-1]
      else:
        kb_path = a
    elif o == '--root-path':
      if not os.path.exists(a):
        print >> sys.stderr, 'Error! Directory ' + a + ' not exists.'
        print >> sys.stderr, 'For help use --help'
        sys.exit(2)
      else:
        root_path = a
    elif o == '--kb-name':
      kb_name = a
    elif o == '--value-k':
      value_k = a
  
  return scenario, lb, ub, feat_def, feat_timeout, discard, kb_path, kb_name, root_path, int(value_k), discardAll


def parse_description(path):
  '''
  Parse the file description.txt of the scenario contained in path. It returns:
    - the list of the algorithms of the scenario;
    - the runtime limit for the scenario;
    - the number of features used in the scenario.
    - the features belonging to each step of the scenario
  '''
  num_features = 0
  pfolio = []
  feature_steps = {}
  timeout = None

  dic = {}
  with open(path + 'description.txt') as stream:
    try:
        dic = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

  # pfolio
  pfolio = dic['metainfo_algorithms'].keys()

  # algorithm_cutoff_time
  timeout = dic['algorithm_cutoff_time']

  # num_features
  if dic['features_deterministic'] != None:
    num_features = len(dic['features_deterministic'])
  if dic['features_stochastic'] != None:
    num_features = num_features + len(dic['features_stochastic'])

  # feature_steps
  for key, value in dic['feature_steps'].iteritems():
    feature_steps[key] = value['provides']

  return pfolio, timeout, num_features, feature_steps

def main(args):
  # Setting variables.
  scenario, lb, ub, feat_def, feat_timeout, discard, kb_path, kb_name, root_path, value_k, discardAll = \
    parse_arguments(args)

  description_path = scenario
  if root_path != '':
    description_path = root_path

  pfolio, timeout, num_features, feature_steps = parse_description(description_path)
  
  if feat_timeout < 0:
    feat_timeout = timeout / 2
  kb_dir = kb_path  + kb_name + '/'
  if not os.path.exists(kb_dir):
    os.makedirs(kb_dir)
  # else:
  #   print >> sys.stderr, 'Warning! Directory ' + kb_dir + ' already exists.'

  # Creating <KB>.info
  writer = csv.writer(open(kb_dir  + kb_name + '.info', 'w'), delimiter = '|')
  # Processing runtimes.
  reader = csv.reader(open(scenario + 'algorithm_runs.arff'), delimiter = ',')
  for row in reader:
    if row and row[0].strip().upper() == '@DATA':
      # Iterates until preamble ends.
      break
  longest_time = 0
  kb = {}
  solved = dict((s, [0, 0.0,0.0]) for s in pfolio)
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

    if info == 'ok' and time > solved[solver][2]:
      solved[solver][2] = time

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
    assert len(feat_vector) == num_features
  
  # Scaling features.
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
    kb_row = [inst, new_feat_vector, kb[inst]]
    writer.writerow(kb_row)


  # Creating <KB>.lims
  lim_file = kb_dir + kb_name + '.lims'
  with open(lim_file, 'w') as outfile:
    json.dump(lims, outfile)
  outfile.close()
  
  selected_features = dict(
      (fn[k], k)
      for k, v in lims.items() 
      if v[0] != v[1] and 
      (not os.path.exists(cost_file) or fn[k] in selected_features)
  )

  step_costs = getAvgFeaturesCost(scenario[:-1])

  # Creating <KB>.args
  neigh_size = int(round(sqrt(len(instances)))) if value_k == -1 else value_k

  if int(timeout) == 0: #incase, missing timeout information
    timeout = 2 * longest_time

  alg_performance = dict(
      (k, v[2])
      for k, v in solved.items()
  )
  print alg_performance
  args = {
    'lb': lb,
    'ub': ub,
    'alg_performance':alg_performance,
    'feat_def': feat_def,
    'backup': backup,
    'timeout': timeout,
    'portfolio': pfolio,
    'neigh_size': neigh_size,
    'static_schedule': [],
    # selected_features[F] is the index of feature F in the original feature 
    # space. (removed feat with a constant value)
    'selected_features': selected_features, # for validation
    'feature_base':selected_features, # for backup
    'feature_wrapper':selected_features, # for learning
    # feature_steps[S] is the list of all the features belonging to step S.
    'feature_steps': feature_steps,
    'step_costs':step_costs
  }
  args_file = kb_dir + kb_name + '.args'
  with open(args_file, 'w') as outfile:
    json.dump(args, outfile)

if __name__ == '__main__':
  main(sys.argv[1:])
   