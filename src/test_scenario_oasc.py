#! /usr/bin/env python

'''
test_scenario [OPTIONS] <SCENARIO_PATH>

Test all the instances of the ASlib scenario according to the SUNNY algorithm. 
For every instance of the scenario, the corresponding SUNNY schedule is printed 
on standard output according to the AS standard format:

  instanceID,runID,solver,timeLimit

Before being tested, a scenario must be trained by using train_scenario script.

Options
=======
  -K <KB_DIR>
   Path of the SUNNY knowledge base. By default, is set to <SCENARIO_PATH>
  -s <STATIC_SCHEDULE>
   Static schedule to be run in the presolving phase for each instance of the 
   scenario. It must be specified in the form: "s_1,t_1,s_2,t_2,...,s_n,t_n"
   meaning that solver s_i has to run for t_i seconds. By default it is empty.
  -k <NEIGH.SIZE>
   The neighbourhood size of SUNNY algorithm. By default, it is set to sqrt(n) 
   where n is the size of the knowledge base.
  -P <s_1,...,s_k>
   The portfolio used by SUNNY. By default, it contains all the algorithms of 
   the scenario.
  -b <BACKUP>
   Sets the SUNNY backup solver. By default, SUNNY uses the Single Best Solver 
   of the portfolio as the backup solver.
  -T <TIMEOUT>
   Sets the timeout of SUNNY algorithm. By default, the scenario cutoff time is 
   used as timeout.
  -o <FILE>
   Prints the predicted schedules to <FILE> instead of std output.
  -f <f_1,...,f_k>
   Specifies the features to be used for solvers prediction. By default, all the 
   features resulting from the training phase (possibly pre-processed) are used.
  -m <MAX-SIZE>
   Maximum sub-portfolio size. By default, it is set to the portfolio size.
  --print-static
   Prints also the static schedule before the dynamic one computed by SUNNY.
   This options is unset by default.
  --help
   Prints this message.
'''
from __future__ import division
import os
import sys
import json
import getopt
from sunny_oasc import *
from tool_feature_steps import necessaryfeatureSteps


def parse_arguments(args):
  '''
  Parse the options specified by the user and returns the corresponding
  arguments properly set.
  '''
  try:
    long_options = ['help', 'print-static']
    opts, args = getopt.getopt(args, 'K:s:k:P:b:T:o:h:f:m:A:', long_options)
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
    
  # Initialize KB variables with default values.
  kb_path = scenario
  kb_name = kb_path.split('/')[-2]

  for o, a in opts:
    if o == '-K':
      if os.path.exists(a):
        kb_path = a
        if kb_path[-1] != '/':
          kb_path += '/'
        kb_name = kb_path.split('/')[-2]
      else:
        print 'Error: ' + a + ' does not exists.'
        print >> sys.stderr, 'For help use --help'
        sys.exit(2)
  # Read arguments.
  args_file = kb_path + kb_name + '.args'
  if not os.path.exists(args_file):
    print >> sys.stderr, 'Error: ' + args_file + ' does not exists.'
    print >> sys.stderr, 'For help use --help'
    sys.exit(2)
  with open(args_file, 'r') as infile:
    args = json.load(infile)
  out_file = None
  new_features = None
  print_static = False
  
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
  step_costs = args['step_costs']
  alg_performance = args['alg_performance']

  max_size = len(portfolio)
  approach = 'default'
  # Options parsing.

  for o, a in opts:
    if o in ('-h', '--help'):
      print(__doc__)
      sys.exit(0)
    elif o == '-s':
      s = a.split(',')
      static_schedule = []
      for i in range(0, len(s) / 2):
        solver = s[2 * i]
        time = float(s[2 * i + 1])
        if time < 0:
          print >> sys.stderr, 'Error! Not acceptable negative time'
          print >> sys.stderr, 'For help use --help'
          sys.exit(2)
        static_schedule.append((solver, time))
    elif o == '-k':
      k = int(a)
    elif o == '-P':
      portfolio = a.split(',')
    elif o == '-b':
      backup = a
    elif o == '-A':
      approach = a
    elif o == '-T':
      timeout = float(a)
    elif o == '-o':
      out_file = a
    elif o == '-f':
      new_features = a.split(',')
    elif o == '-m':
      max_size = int(a)
      if max_size < 1 or max_size > len(portfolio):
        print >> sys.stderr, 'Error! Not acceptable size'
        print >> sys.stderr, 'For help use --help'
        sys.exit(2)
    elif o == '--print-static':
      print_static = True
        
  
  selfeats = selected_features.keys()
  # print scenario
  stepOrder = necessaryfeatureSteps(scenario[:-1], selfeats)
  
  # print scenario
  return k, lb, ub, feat_def, kb_path, kb_name, static_schedule, timeout,      \
    portfolio, backup, out_file, scenario, print_static, selected_features,    \
      feature_steps, max_size,step_costs,stepOrder,approach,alg_performance

def schedule_post_process(schedule,alg_performance):
  schedulex = []
  for indx in range(0,len(schedule)):
    alg = schedule[indx][0]
    best_run = alg_performance[alg]*5 
    runtime = schedule[indx][1] if best_run > schedule[indx][1] else best_run
    schedulex.append([alg,runtime])

  schedulex[-1] = schedulex[-1][0]
  return schedulex

def main(args):
  k, lb, ub, feat_def, kb_path, kb_name, static_schedule, timeout, portfolio,  \
    backup, out_file, scenario, print_static, selected_features, feature_steps,\
      max_size,step_costs,stepOrder,approach,alg_performance = parse_arguments(args)
  
  # print selected_features
  # print stepOrder
  # print step_costs

  total_cost = 0
  if step_costs:
    total_cost = sum(
        cost
        for (step, cost) in step_costs.items() 
        if step in stepOrder
    )

  reader = csv.reader(open(scenario + 'feature_values.arff'), delimiter = ',')
  for row in reader:
    if row and row[0].strip().upper() == '@DATA':
      # Iterates until preamble ends.
      break
  header = 'instanceID,runID,solver,timeLimit'
  if out_file:
    writer = csv.writer(open(out_file, 'w'), delimiter = ',')
    writer.writerow(header.split(','))
  else:
    print header

  number_solver = 0
  instancecount = 0
  employed_solvers = []
  rlt_oasc = {}
  for row in reader:
    if len(row) < 2:
      continue # un regular data
    inst = row[0]
    feat_vector = row[2:]

    # Get the schedule computed by SUNNY algorithm.
    schedule = get_sunny_schedule(
      lb, ub, feat_def, kb_path, kb_name, static_schedule, timeout, k, \
      portfolio, backup, selected_features, feat_vector, total_cost, max_size
    )

    # print schedule
    instancecount = instancecount + 1
    number_solver +=  len(schedule)
    inst_employ_solvers = ''

    i = 1
    if print_static:
      schedule = static_schedule + schedule
      i = 0
    for (s, t) in schedule:
      row = inst + ',' + str(i) + ',' + s + ',' + str(t)
      inst_employ_solvers += s + ':' + str(t) + ','
      if out_file:
        writer.writerow(row.split(','))
      else:
        print row
      i += 1

    # print schedule
    # free time budget for backup solver 
    schedule = schedule_post_process(schedule,alg_performance)

    # print [inst,stepOrder+schedule]
    rlt_oasc[inst] = stepOrder + schedule
    employed_solvers.append(inst_employ_solvers)

  execution_log = {}
  execution_log['number_solver'] = round(float(number_solver/instancecount),2) 
  execution_log['employed_solvers'] = employed_solvers 
  execution_log['k'] = k 
  
  with open(kb_path+'execution_log.json', 'w') as outfile:
     json.dump(execution_log, outfile, sort_keys = True, indent = 4,ensure_ascii=False)

  # output result
  root_arr = os.path.realpath(__file__).split('/')[:-2]
  root = '/'.join(root_arr) 
  scenarioname = scenario.split("/")[-2]

  result_dir = root + '/results/'+approach
  result_path = result_dir+'/'+scenarioname+'.json'
  print 'Results stored in: ',result_path
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)

  with open(result_path, 'w') as outfile:
    json.dump(rlt_oasc, outfile)    

if __name__ == '__main__':
  main(sys.argv[1:])