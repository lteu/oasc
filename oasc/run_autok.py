#! /usr/bin/env python

'''
python run_autok.py <SCENARIO_NAME>


comments
========
coordinate macro learning process

Author: hearntest
'''

import os
import sys
import datetime
import time
from kit import *
from run_fkvar import learn_scenario_fold

#=======================================================
#=======================================================
#=======================================================

def autok_learning(param_learn):
  scenario_path,scenario_cv,scenario_outcome_dir,tdir,src_path,kRange,lim_nfeat,shouldWrappeFeature = parse_param_learn(param_learn)

  start_time_fold = time.time()

  sub_scenario_path = scenario_cv

  print 'Building context ...'
  context = build_context(sub_scenario_path)

  log_file_hard,log_file_small,outcome_file = initFiles(scenario_outcome_dir,kRange[0],kRange[-1])
  features = informative_feat(sub_scenario_path+'/'+tdir)

  param_learn_fold = {
    'sub_scenario_path':sub_scenario_path,
    'log_file_hard':log_file_hard,
    'log_file_small':log_file_small,
    'outcome_file':outcome_file,
    'features':['']
  }

  # learn k values
  best_feats,best_k,par10pre = learn_scenario_fold(param_learn,param_learn_fold,context,1)

  final_k =  best_k
  final_feat = best_feats
  final_par10 =  par10pre

  time_diff = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time_fold)) 

  date_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  appendToFile(outcome_file,'train_1_'+'; k='+str(final_k)+'; feats='+final_feat+'; par10='+str(final_par10)+'; runtime='+time_diff+'; '+str(date_now)+"\n\n")

  print 'FINAL RESULT:',final_par10,final_feat,final_k,time_diff


def main(args):
  if len(args) == 0:
    sys.exit('Missing Arg, E.g. Scenario Name ...')

  scenario_name = args[0]

  tdir = 't'
  lim_nfeat = 300 # trivial
  outcomeDirname = 'outcome-autok'

  scenario_path,scenario_cv,src_path = prepare_scenario(scenario_name,tdir,-1)

  param_learn = {
    'scenario_path':scenario_path,
    'scenario_cv':scenario_cv,
    'scenario_outcome_dir':scenario_path+"/"+outcomeDirname,
    'tdir':tdir,
    'src_path':src_path,
    'kRange':range(3,81),
    'lim_nfeat':lim_nfeat,
    'shouldWrappeFeature':False
  }
  autok_learning(param_learn)

if __name__ == '__main__':
  main(sys.argv[1:])
  