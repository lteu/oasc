#! /usr/bin/env python

'''
Tool Kit for learning module


comments
========
support for main.py

Author: hearntest
'''


import os
import csv
import sys
import json
import random
import datetime
import time
from shutil import copyfile
# import sys
root_arr = os.path.realpath(__file__).split('/')[:-2]
root = '/'.join(root_arr) 
src_path = root + '/src/'
sys.path.append(src_path)

from train_scenario_oasc import parse_description
from sunny_mock import pre_process_mock,test_scenario_mock,train_scenario_mock,getFeatureCost
from subprocess import Popen
from math import sqrt

from analyze_scenario import analyze_scenario_stats,getRunTime



def learn_scenario_fold(param_learn,param_learn_fold,context,fist_n_ks):
  
  scenario_path,scenario_cv,scenario_outcome_dir,tdir,src_path,kRange,lim_nfeat,shouldWrappeFeature= parse_param_learn(param_learn)
  sub_scenario_path,log_file_hard,log_file_small,outcome_file,features = parse_param_learn_fold(param_learn_fold)
  
  start_time_fold = time.time()
  # iteration for best configurators
  gain = best_k = new_gain = best_par10 = float('inf')
  best_feats = ''
  
  while True:
    new_sel_feat = ''
    new_best_k = -1
    new_gain = float('inf')
    tmp_par10 = None
    print len(features), 'feature space size'
    for feat in features:
      
      feat_str = feat

      testFeatures = feat_str if best_feats == '' else best_feats+','+feat_str

      # check saved states (if computed yet)
      if "feats="+testFeatures+";" in open(log_file_small).read():
        print "feats=",testFeatures,"; already computed" 
        token = "feats="+testFeatures+";"
        par10,value_k = read_state_file(log_file_small,token)
      else:
  
        # start main
        start_time_feat = time.time()

        # MAIN
        tmp_par10,value_k,par10s = learn_optima_k(src_path,sub_scenario_path,kRange,testFeatures,log_file_hard,context)
        par10 = sum(par10s[:fist_n_ks]) # top 3 par10 as final score

        print par10,value_k,testFeatures
        # save state
        log_small(log_file_small,start_time_feat,testFeatures,value_k,par10)

      if par10 < new_gain:
        new_gain = par10
        new_best_k = value_k
        new_sel_feat = feat_str
        best_par10 = tmp_par10
    # end for feat in features

    if new_gain > gain:
      print new_gain,'>', gain,' so break, final par10',gain
      break

    # pos process
    gain = new_gain
    best_k = new_best_k
    best_feats = new_sel_feat if best_feats == '' else best_feats+','+new_sel_feat
    features = [i for i in features if str(i) != str(new_sel_feat)] # remove selected feature from feature set

    # debug log
    log_small_subtotal(log_file_small,best_feats,best_k,gain)

    number_of_sel_feats = len(best_feats.split(','))

    # uncomment here for limited features
    if number_of_sel_feats == lim_nfeat:
      print lim_nfeat,'features reached, so break, final par10:',gain
      break
  
  # in case resumed from previous state, recompute best value
  if not best_par10:
    best_par10,value_k,par10s = learn_optima_k(src_path,sub_scenario_path,kRange,best_feats,log_file_hard,context)
    # print value_k,best_k,' these two values should be the same ...'
    
  # end while
  return best_feats,best_k,best_par10

def learn_optima_k(src_path,sub_scenario_path,kRange,testFeatures,log_file_hard,context):
  low_par10 = 999999
  best_k = -1

  par10s = []
  for value_k in kRange:
    start_time_k = time.time()

    params = {'k':value_k,'feat':testFeatures}

    token = "k="+str(value_k)+";feats="+testFeatures+";"
    if token in open(log_file_hard).read():

      print token,"already computed" 
      par10,value_k = read_state_file(log_file_hard,token)

    else:

      # print params
      par10 = run_evaluator(src_path,sub_scenario_path,params,context)

      ex_time_k = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_k)) 
      date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      log_str = 'k='+str(value_k)+";feats="+testFeatures+";par10="+str(par10)+';runetime='+ex_time_k+';'+str(date_now)+"\n"
      appendToFile(log_file_hard,log_str)
    
    par10s.append(par10)
    if low_par10 > par10:
      low_par10 = par10
      best_k = value_k

  par10s.sort()
  return low_par10,best_k,par10s


# run with feedback
def run_evaluator(src_path,scenario_path,params,context):
  '''
  Interface, execute SUNNY like evaluation algorithm
  if test with all features, then set params['feat'] = '' 
  '''
  d_date = datetime.datetime.now()
  reg_format_date = d_date.strftime("%Y-%m-%d-%H-%M-%S")

  cv_train = 'cv_'+scenario_path.split('/')[-1]

  testK = params['k']
  testFeat = params['feat']
  
  # print 'testing: k',testK,' feat',testFeat, 'score',

  par10 = 0
  cv_folders = context['cv_folders']
  for subdir in cv_folders:
    case_subdir = scenario_path + '/'+cv_train+'/'+subdir+'/'
    test_dir = scenario_path + '/'+cv_train+'/'+ subdir.replace('train_', 'test_') + '/'
    kbs = []
    kbs = context['kb'][case_subdir]
    instt = context['test'][0][test_dir]
    fcost = context['test'][1][test_dir]
    test = [instt,fcost]
    ground = context['ground'][test_dir]
    par10 += small_sunny(src_path,scenario_path,case_subdir,test_dir,params,kbs,test,ground)

  return par10 

# find k values
def k_par10_scenario(src_path,kRange,sub_scenario_path,context):
  rlt = []
  for value_k in kRange:
    start_time_k = time.time()
    print 'calculating k',value_k,
    params = {'k':value_k,'feat':''}
    par10 = run_evaluator(src_path,sub_scenario_path,params,context)
    print par10
    rlt.append([value_k,par10])

  return rlt

def small_sunny(src_path,scenario_path,case_subdir,test_dir,params,kbs,test,ground):
  '''
  SUNNY Gready Version
  '''
  par10 = 0
  testK = params['k']
  testFeat = params['feat']

  kb_name = case_subdir.split('/')[-1]

  pred_file = test_dir + '/predictions.csv'

  args = kbs[0]
  info = kbs[1]
  lims = kbs[2]
  
  if str(testK) != '':
    args['neigh_size'] = int(testK)
  else:
    args['neigh_size'] = int(round(sqrt(len(info))))

  # feature selection

  if str(testFeat) != '':
    options = [
      '-E', 'wrapper',
      '-S', testFeat,
      '--filter-portfolio'
    ]
    input_args = options + case_subdir.split()
    args = pre_process_mock(input_args,args)

  # prediction

  input_args = ('-o ' + pred_file \
  + ' --print-static -K ' + case_subdir + '/kb_' + kb_name + ' ' + test_dir).split(' ')

  par10 = test_scenario_mock(input_args,test_dir,args,lims,info,test,ground)  

  return par10

  
# Original SUNNY
# ==========================

# def run_sunny(src_path,scenario_path,params):
#   '''
#   Original SUNNY
#   '''
#   d_date = datetime.datetime.now()
#   reg_format_date = d_date.strftime("%Y-%m-%d-%H-%M-%S")

#   cv_train = 'cv_'+scenario_path.split('/')[-1]

#   testK = params['k']
#   testFeat = params['feat']
#   print 'k',testK,' feat',testFeat

#   for subdir in os.listdir(scenario_path + '/' + cv_train):
#     if 'train_' in subdir and 'kb_' not in subdir:
#       case_subdir = scenario_path + '/'+cv_train+'/'+subdir
    
#       test_dir = scenario_path + '/'+cv_train+'/'+ subdir.replace('train_', 'test_')

#       call_sunny(src_path,scenario_path,case_subdir,test_dir,params)

#   avgfsi_sunny, avgfsi_vbs, avgpar10_sunny, avgpar10_vbs,m,n,scenario,avgfsi_sbs,avgpar10_sbs = analyze_scenario_stats(scenario_path)
  
#   return avgpar10_sunny 


def run_sunny_fold(src_path,scenario_path,params,fold):
  '''
  Original SUNNY per fold 
  compute and save results in disk files
  '''
  cv_train = 'cv_'+scenario_path.split('/')[-1]
  case_subdir = scenario_path + '/'+cv_train+'/'+'train_1_'+str(fold)
  test_dir = scenario_path + '/'+cv_train+'/'+'test_1_'+str(fold)
  call_sunny(src_path,scenario_path,case_subdir,test_dir,params)
  
    
def run_sunny_oasc(src_path,scenario_test,scenario_train,params,approach):
  '''
  Original SUNNY per fold 
  compute and save results in disk files
  '''

  testK = params['k']
  testFeat = params['feat']

  kb_name = scenario_train.split('/')[-1]
  pred_file = scenario_test + '/predictions.csv'

  options = ' --discard --feat-timeout +inf '
  options += '--root-path '+scenario_train+'/ '

  if str(testK) != '':
    options += '--value-k '+str(testK) + ' '
  
  cmd = 'python ' + src_path + 'train_scenario_oasc.py ' + options + scenario_train
  proc = Popen(cmd.split())
  proc.communicate()
  
  description_path = scenario_train+'/'
  pfolio, timeout, num_features, feature_steps, maximize = parse_description(description_path)

  if str(testFeat) != '':
    options = [
      '--kb-path', scenario_train + '/kb_' + kb_name,
      '-E', 'wrapper',
      '-S', testFeat
    ]
    if not maximize: 
    # scenarios to maximize are special, leave unfilter portfolio at this moment
      options.append('--filter-portfolio')

    cmd = ['python', src_path + 'pre_process.py'] + options + scenario_train.split()
    proc = Popen(cmd)
    proc.communicate()
  
  options = ' ' 
  cmd = 'python ' + src_path + 'test_scenario_oasc.py ' + options + ' -o ' + pred_file \
  +' -A '+approach +' --print-static -K ' + scenario_train + '/kb_' + kb_name + ' ' + scenario_test 

  proc = Popen(cmd.split())
  proc.communicate()
  
      

def call_sunny(src_path,scenario_path,case_subdir,test_dir,params):
  '''
  Original SUNNY implementation
  '''
  testK = params['k']
  testFeat = params['feat']

  kb_name = case_subdir.split('/')[-1]
  pred_file = test_dir + '/predictions.csv'

  options = ' --discard --feat-timeout +inf '
  options += '--root-path '+scenario_path+'/ '

  if str(testK) != '':
    options += '--value-k '+str(testK) + ' '

  cmd = 'python ' + src_path + 'train_scenario.py ' + options + case_subdir
  proc = Popen(cmd.split())
  proc.communicate()


  if str(testFeat) != '':
    options = [
      '--kb-path', case_subdir + '/kb_' + kb_name,
      '-E', 'wrapper',
      '-S', testFeat,
      '--filter-portfolio'
    ]
    cmd = ['python', src_path + 'pre_process.py'] + options + case_subdir.split()
    proc = Popen(cmd)
    proc.communicate()

  options = ' ' 
  cmd = 'python ' + src_path + 'test_scenario.py ' + options + ' -o ' + pred_file \
  + ' --print-static -K ' + case_subdir + '/kb_' + kb_name + ' ' + test_dir 

  proc = Popen(cmd.split())
  proc.communicate()


# ==============================
# Initial Scenarios Preparation
# ==============================


def prepare_scenario(scenario_name,tdir,number_insts):
  '''
  Initial Train and Split

  '''
  root_arr = os.path.realpath(__file__).split('/')[:-2]
  root = '/'.join(root_arr) 
  src_path = root + '/src/'

  scenario_path = root + '/data/oasc_scenarios/train/'+scenario_name
  scenario_cv = scenario_path
  scenario_t = scenario_path + '/'+tdir

  if not os.path.exists(scenario_path):
    print 'Scenario Name Err on',scenario_path
    sys.exit()

  if not os.path.exists(scenario_t):
    print 'Creating folder t'
    copyTrainFiles(scenario_path,tdir) 

  # split train
  print 'Check and Split Training Data ...'

  smart_conf = scenario_path+"/smart.txt"
  shouldSplit = False
  if not os.path.exists(smart_conf):
    shouldSplit = True
  else:
    with open(smart_conf, 'r') as f:
      first_line = f.readline()
      if first_line == '' or int(first_line) != number_insts:
        shouldSplit = True

  subdir = scenario_name
  cv_subdir = scenario_cv
  if not os.path.exists(cv_subdir+'/cv_'+subdir) or shouldSplit:
    case_subdir = cv_subdir+'/t'
    options = ' --discard --feat-timeout +inf '
    options += '--root-path '+scenario_path+'/ '
    input_args = (options + case_subdir).split(' ')
    input_args = [x for x in input_args if x]

    kb_arg,kb_info,kb_lims = train_scenario_mock(input_args)
    with open(case_subdir+'/kb.args', 'w') as outfile:
      json.dump(kb_arg, outfile)
    smart_split(cv_subdir,kb_info,number_insts)
        
  if shouldSplit:              
    with open(smart_conf,'w') as f:
      f.write(str(number_insts))    

  return scenario_path,scenario_cv,src_path

# ==========================
# Splitting Scenarios
# ==========================

def copyTrainFiles(path,tdir):
  directory = path+'/'+tdir
  if not os.path.exists(directory):
    os.makedirs(directory)
  alg = path+'/algorithm_runs.arff'
  val = path+'/feature_values.arff'
  cst = path+'/feature_costs.arff'
  t_alg = directory+'/algorithm_runs.arff'
  t_val = directory+'/feature_values.arff'
  t_cst = directory+'/feature_costs.arff'
  copyfile(val, t_val)
  copyfile(alg, t_alg)
  if os.path.exists(cst):
    copyfile(cst, t_cst)


def smart_split(path,kb_path,inst_k): 
  '''
  Split top k ordered instances 
  (tested only for sub-sub directory)
  '''
  instance_arr = top_inst_bk(kb_path,inst_k)
  createCV(path,instance_arr)


def random_split(path,tdir):
  '''
  Split all the instances
  '''
  instance_arr = []
  instance_reader = csv.reader(open(path+'/'+tdir+'/feature_values.arff'), delimiter = ',')
  for row in instance_reader:
    if row and row[0].strip().upper() == '@DATA':
      break
  for row in instance_reader:
    instance_arr.append(row[0])
  createCV(path,instance_arr)

def top_inst_bk(kbinfo,inst_k):
  '''
  support for smart split
  get top k  instances ordered by their importance
  '''
  # if not os.path.exists(kbInfoFile):
  #   print 'Cannot find',kbInfoFile
  #   sys.exit()
  
  solver_insts = {}
  # with open(kbInfoFile) as f:
  #   lines = f.readlines()
  lines = kbinfo
  insts = []
  for line in lines:

    # pieces = line.split("|")
    pieces = line
    solverinfo =  pieces[2]
    dic = solverinfo
    item =  [pieces[0],dic]
    insts.append(item)

  for ins in insts:
    ins_id = ins[0]
    best_solver =''
    best_time = float("inf")
    success_count = 0

    for (solver,info) in ins[1].items():
      run_info = info['info']
      time_info = float(info['time'])
      if run_info == 'ok':
        success_count += 1
      if time_info < best_time:
        best_time = time_info
        best_solver = solver

    sol_ins = [ins_id,success_count]
    if best_solver in solver_insts:
      solver_insts[best_solver].append(sol_ins)
    else:
      solver_insts[best_solver] = [sol_ins]

  for (solver,instances) in solver_insts.items():
    sortedinsts = sorted(instances, key=lambda x: x[1])
    solver_insts[solver] = sortedinsts
    
  best_insts = []

  if inst_k == -1:
    inst_k = len(insts)

  while len(best_insts) < inst_k:
    bf = len(best_insts)
    for (solver,instances) in solver_insts.items():
      if len(instances) > 0:
        best_insts.append(instances.pop(0))

    # avoid dead loop
    if len(best_insts) == bf:
      break

  sel_insts =  [item[0] for item in best_insts]
  return sel_insts

def partition_shuffle(lst, n): 
  '''
  Random partition
  '''
  random.shuffle(lst)
  division = len(lst) / float(n) 
  return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]



def chunkify(lst,n):
  '''
  equal partition
  http://stackoverflow.com/questions/2130016/splitting-a-list-of-arbitrary-size-into-only-roughly-n-equal-parts
  '''
  return [ lst[i::n] for i in xrange(n) ]

def partition(lst, n): 
  '''
  division partition
  '''
  division = len(lst) / float(n) 
  return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

def createCV(path,instance_arr):
  '''
  create Cross Validaton files
  '''
  number_of_folds = 10
  cv_train = 'cv_'+path.split('/')[-1]
  folder = path + '/' +cv_train

  if not os.path.exists(folder):
    os.makedirs(folder)
  else:
    print 'Creating CV, following directory will be overwrited' + folder

  # spliting: create ten folds, other options: partition, partition_shuffle
  folds = chunkify(instance_arr,number_of_folds)
  for idx_cv in xrange(0,number_of_folds):
    train_dir = folder + '/' + 'train_'+ str(idx_cv) + '/'
    test_dir = folder + '/' + 'test_'+ str(idx_cv) + '/'

    if not os.path.exists(train_dir):
      os.makedirs(train_dir)
    if not os.path.exists(test_dir):
      os.makedirs(test_dir)

    for infile in ['algorithm_runs.arff', 'feature_values.arff', 'feature_costs.arff']:
      # print path ,'/',infile
      if not os.path.exists(path +'/t/'+infile):
        continue

      reader = csv.reader(open(path +'/'+infile), delimiter = ',')
      writer_train = csv.writer(open(train_dir + infile, 'w'))
      writer_test = csv.writer(open(test_dir + infile, 'w'))
      for row in reader:
        writer_train.writerow(row)
        writer_test.writerow(row)
        if row and row[0].strip().upper() == '@DATA':
          # Iterates until preamble ends.
          break
      for row in reader:
        if row[0] in folds[idx_cv]:
          writer_test.writerow(row)
        elif row[0] in instance_arr:
          writer_train.writerow(row)

# ==========================
# Training Scenarios: KBs
# ==========================

def build_context(sub_scenario_path):
  kbs_dic = {}
  kbs_dic = make_train_kbs(sub_scenario_path)
  instt_dic,fcost_dic = make_test_kbs(sub_scenario_path,kbs_dic)
  runtime_dic = make_ground_kbs(sub_scenario_path)
  cv_folders = get_cv_folders(sub_scenario_path)
  context = {'kb':kbs_dic,'test':[instt_dic,fcost_dic],'ground':runtime_dic,'cv_folders':cv_folders}
  return context

def make_train_kbs(scenario_path):
  '''
  Create trained KBs per sub-fold
  '''
  kbs_dic = {}
  cv_train = 'cv_'+scenario_path.split('/')[-1]

  for subdir in os.listdir(scenario_path + '/' + cv_train):
    if 'train_' in subdir and 'kb_' not in subdir:
      case_subdir = scenario_path + '/'+cv_train+'/'+subdir+'/'
    
      options = ' --discard --feat-timeout +inf '
      options += '--root-path '+scenario_path+'/ '

      # train scenario

      kb_name = case_subdir.split('/')[-1]
      input_args = (options + case_subdir).split(' ')
      input_args = [x for x in input_args if x]

      args,info,lims = train_scenario_mock(input_args)
      kbs_dic[case_subdir] = [args,info,lims]


  return kbs_dic

def make_test_kbs(scenario_path,kbs_dic):
  '''
  Create test KBs per sub-fold
  '''
  fcost_dic = {}
  instt_dic = {}
  cv_train = 'cv_'+scenario_path.split('/')[-1]
  for subdir in os.listdir(scenario_path + '/' + cv_train):
    if 'test_' in subdir and 'kb_' not in subdir:
      
      test_path = scenario_path + '/'+cv_train+'/'+subdir+'/'
      train_path = scenario_path + '/'+cv_train+'/'+subdir.replace('test_', 'train_')+'/'
      args = kbs_dic[train_path][0]
      feature_steps = args['feature_steps']
      feature_costs = getFeatureCost(test_path,feature_steps)
      fcost_dic[test_path] =  feature_costs

      insts = []
      reader = csv.reader(open(test_path + 'feature_values.arff'), delimiter = ',')
      for row in reader:
        if row and row[0].strip().upper() == '@DATA':
          # Iterates until preamble ends.
          break

      for row in reader:
        inst = row[0]
        feat_vector = row[2:]
        insts.append([inst,feat_vector])

      instt_dic[test_path] = insts

  return instt_dic,fcost_dic

def make_ground_kbs(scenario_path):
  '''
  Create test ground sets per sub-fold
  '''
  runtime_dic = {}
  cv_train = 'cv_'+scenario_path.split('/')[-1]
  for subdir in os.listdir(scenario_path + '/' + cv_train):
    if 'test_' in subdir and 'kb_' not in subdir:
      test_path = scenario_path + '/'+cv_train+'/'+subdir+'/'
      runtimes = getRunTime(test_path)
      runtime_dic[test_path] =  runtimes

  return runtime_dic

def informative_feat(path):
  ''' 
  get features (useful) space for Wrapper
  '''
  args_path = path+'/kb.args'
  if  os.path.exists(args_path):
    with open(args_path) as infile:
      args = json.load(infile)
  else:
    sys.exit('Err',args_path,' cannot be found')

  count = 0
  fArr = []
  for feat_tuple in args['selected_features'].items():
    fArr.append(feat_tuple[0])
    count = count + 1
  return fArr

def get_cv_folders(scenario_path):
  '''
  get sub-fold list
  '''
  folders = []
  cv_train = 'cv_'+scenario_path.split('/')[-1]
  for subdir in os.listdir(scenario_path + '/' + cv_train):
    if 'train_' in subdir and 'kb_' not in subdir:
      folders.append(subdir)
  return folders

# =================================
# Dictionary Parser
# =================================

def parse_param_learn(param_learn):
  scenario_path = param_learn['scenario_path']
  scenario_cv = param_learn['scenario_cv']
  scenario_outcome_dir = param_learn['scenario_outcome_dir']
  tdir = param_learn['tdir']
  src_path = param_learn['src_path']
  kRange = param_learn['kRange']
  lim_nfeat = param_learn['lim_nfeat']
  shouldWrappeFeature = param_learn['shouldWrappeFeature']
  return scenario_path,scenario_cv,scenario_outcome_dir,tdir,src_path,kRange,lim_nfeat,shouldWrappeFeature

def parse_param_learn_fold(param_learn_fold):
  sub_scenario_path = param_learn_fold['sub_scenario_path']
  log_file_hard = param_learn_fold['log_file_hard']
  log_file_small = param_learn_fold['log_file_small']
  outcome_file = param_learn_fold['outcome_file']
  features = param_learn_fold['features']
  return sub_scenario_path,log_file_hard,log_file_small,outcome_file,features

# =================================
# State Record & Resumable Support
# =================================

def read_state_file(log_file_small,token):
  '''
  Read computed params for Resumable Support
  '''
  with open(log_file_small,"r") as fi:
    for ln in fi:
        if  token in ln:
          pieces = ln.split(';')
          for pie in pieces:
            if 'par10=' in pie:
              par10 = float(pie.split('=')[1])
            if 'k=' in pie:
              value_k = pie.split('=')[1]
  return par10,value_k

def initFiles(scenario_outcome_dir,low_k,high_k):
  ''' 
  create files PER FOLD if not exist, does NOT override 
  '''
  if not os.path.exists(scenario_outcome_dir):
    os.makedirs(scenario_outcome_dir)

  log_file_hard = scenario_outcome_dir+'/log-hard'+'-k-'+str(low_k)+'-'+str(high_k)+'.txt'
  log_file_small = scenario_outcome_dir+'/log-small'+'-k-'+str(low_k)+'-'+str(high_k)+'.txt'
  outcome_file = scenario_outcome_dir+'/outcome'+'-k-i-i.txt'
  files = [log_file_hard,log_file_small,outcome_file]
  for filepath in files:
    if not os.path.exists(filepath):
      file(filepath, 'w+').close()
  return log_file_hard,log_file_small,outcome_file
  
def writeToFile(filepath,content):
  ''' 
  output content to file 
  '''
  with open(filepath, 'w+') as the_file:
    the_file.write(content)

def appendToFile(filepath,content):
  ''' 
  append text to file content
  '''
  with open(filepath, 'a') as the_file:
    the_file.write(content)

# Interfaces
# --------------

def log_hard(log_file_hard,features):
  num_of_features = len(features)
  features_in_str = json.dumps(features)
  appendToFile(log_file_hard,'\ntrain_1'+'\n------------\n'+str(num_of_features)+'features\n'+features_in_str+'\n') 

def log_small(log_file_small,start_time_feat,testFeatures,value_k,par10):
  ex_time_feat = str(datetime.timedelta(seconds=(time.time() - start_time_feat))) 
  date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  log_str = "feats="+str(testFeatures)+";k="+str(value_k)+';par10='+str(par10)+";runtime="+ex_time_feat+';'+str(date_now)+"\n"
  appendToFile(log_file_small,log_str)

def log_small_subtotal(log_file_small,best_feats,best_k,gain):
  log_str_subtotal = '------------ '+"feats="+str(best_feats)+"- k="+str(best_k)+"- "+'par10='+str(gain)+' ------------'
  if not log_str_subtotal in open(log_file_small).read():
    appendToFile(log_file_small,"\n"+log_str_subtotal+"\n")

# Math Tools
# --------------

def minmax_scale(data, feature_range):
  data = [ float(elem) for elem in data ]
  a = feature_range[0]
  b = feature_range[1]
  minv = min(data)
  maxv = max(data)
  data = [ (b-a)*(elem-minv)/(maxv-minv) + a for elem in data ]
  data = [ round(elem, 1) for elem in data ]  
  return data

def best_n_k(arr,n):
  # print arr
  arr = [ [int(elem[0]),float(elem[1])] for elem in arr ]
  data = [ float(elem[1]) for elem in arr ]
  data = minmax_scale(data, feature_range=(1,10))
  print data
  newarr = []
  for idx in range(len(data)):
    newarr.append([arr[idx][0],data[idx]])
  newarr.sort(lambda x,y : cmp(x[1], y[1]))
  newarr = [i[0] for i in newarr if i[0]>3][:n]
  return newarr
