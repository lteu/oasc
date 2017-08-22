#! /usr/bin/env python

'''
python run_fkvar.py <SCENARIO_NAME>


comments
========
coordinate macro training process

Author: hearntest
'''

import os
import sys
import datetime
import time
from kit import *



def learnK_with_all_feats(param_learn,param_learn_fold,context):
  scenario_path,scenario_cv,scenario_outcome_dir,tdir,src_path,kRange,lim_nfeat,shouldWrappeFeature= parse_param_learn(param_learn)
  
  sub_scenario_path,log_file_hard,log_file_small,outcome_file,features = parse_param_learn_fold(param_learn_fold)
  
  start_time_fold = time.time()

  par10,value_k,par10s = learn_optima_k(src_path,sub_scenario_path,kRange,'',log_file_hard,context)

  date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

  ex_time_fold = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_fold)) 
  
  appendToFile(outcome_file,'train_1_; k='+str(value_k)+'; feats=all; par10='+str(par10)+'; runtime='+ex_time_fold+'; '+str(date_now)+"\n\n")
  
  log_small(log_file_small,start_time_fold,'all',value_k,par10)

  return value_k,par10



def learn_scenario(param_learn):
  scenario_path,scenario_cv,scenario_outcome_dir,tdir,src_path,kRange,lim_nfeat,shouldWrappeFeature = parse_param_learn(param_learn)

  start_time_fold = time.time()

  sub_scenario_path = scenario_cv

  print 'Building context ...'

  context = build_context(sub_scenario_path)

  # create resumable state
  log_file_hard,log_file_small,outcome_file = initFiles(scenario_outcome_dir,kRange[0],kRange[-1])
  features = informative_feat(sub_scenario_path+'/'+tdir) if shouldWrappeFeature else ['']
  log_hard(log_file_hard,features)

  param_learn_fold = {
    'sub_scenario_path':sub_scenario_path,
    'log_file_hard':log_file_hard,
    'log_file_small':log_file_small,
    'outcome_file':outcome_file,
    'features':features
  }

  best_feats,best_k,par10f = learn_scenario_fold(param_learn,param_learn_fold,context,3)

  print 'Step 1: ',par10f, best_k, best_feats,

  time_diff = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_fold)) 

  date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  appendToFile(outcome_file,'train_1_; k='+str(best_k)+'; feats='+best_feats+'; par10='+str(par10f)+'; runtime='+time_diff+'; '+str(date_now)+"\n\n")

  # BACKUP APPROACH
  print 'Learning K with all the features'

  # if not "feats=all;" in open(log_file_small).read():
  param_learn['kRange'] = range(3,81)
  value_k,par10k = learnK_with_all_feats(param_learn,param_learn_fold,context)
  # else:
  #   token = "feats=all;"
  #   par10k,value_k = read_state_file(log_file_small,token)

  print 'Step 2: ',par10k, value_k

  # compare par10 of different approaches
  final_k = value_k if par10k < par10f else best_k
  final_feat = 'all' if par10k < par10f else best_feats
  final_par10 = par10k if par10k < par10f else par10f

  # final_k = best_k
  # final_feat = best_feats
  # final_par10 = par10f
  time_diff = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_fold)) 

  date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  
  appendToFile(outcome_file,'train_1_'+'; k='+str(final_k)+'; feats='+final_feat+'; par10='+str(final_par10)+'; runtime='+time_diff+'; '+str(date_now)+"\n\n")

  print 'FINAL RESULT:',final_par10,final_feat,final_k,time_diff

#=======================================================
#=======================================================
#=======================================================

def main(args):
  if len(args) == 0:
    sys.exit('Missing Arg, E.g. Scenario Name ...')

  scenario_name = args[0]

  # configuration settings
  lim_nfeat = 5
  tdir = 't'
  outcomeDirname = 'outcome-fkvar'

  low_k = 3
  high_k = 30
  kRange = range(low_k,high_k)
  
  # less insts make training faster
  # -1 means smart split that takes all the dataset, other integer means taking that amount of insts
  scenario_path,scenario_cv,src_path = prepare_scenario(scenario_name,tdir,1500)

  param_learn = {
  'scenario_path':scenario_path,
  'scenario_cv':scenario_cv,
  'scenario_outcome_dir':scenario_path+"/"+outcomeDirname,
  'tdir':tdir,
  'src_path':src_path,
  'kRange':kRange,
  'lim_nfeat':lim_nfeat,
  'shouldWrappeFeature':True
  }
  learn_scenario(param_learn)



if __name__ == '__main__':
  main(sys.argv[1:])
  