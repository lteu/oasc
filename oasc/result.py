#! /usr/bin/env python

'''

python result.py <SCENARIO_NAME> <Training-Approach>

eg: 
$ python result.py Caren autok
$ python result.py Caren fkvar

comments
========
test scenarios with calculated params

Author: HearNest

'''

import os
import csv
import sys
from kit import *

def getTrainingOutcome(outcome_path):
  '''
  retrive trained params
  '''
  with open(outcome_path, 'r') as f:
    lines = f.readlines()
    lines = [elem for elem in lines if elem.strip() != '']
    if len(lines) == 0:
      sys.exit(outcome_path+' is empty, please train this scenario first')
      
    last_line = lines[-1]
    if last_line.strip() == '':
      sys.exit('File might be empty'+outcome_path)
    pieces = last_line.split(";")
    value_k =  pieces[1].split("=")[1]
    testFeatures =  pieces[2].split("=")[1]
  return value_k,testFeatures


def validate_trained_scenario(scenario_name,postfix):
  '''
  test scenarios
  '''
  root_arr = os.path.realpath(__file__).split('/')[:-2]
  root = '/'.join(root_arr) 
  src_path = root + '/src/'

  scenario_test = root + '/data/oasc_scenarios/test/' + scenario_name
  scenario_train = root + '/data/oasc_scenarios/train/' + scenario_name

  outcomedirpath = scenario_train+'/outcome-'+postfix+'/'
  if not os.path.exists(outcomedirpath):
    sys.exit(outcomedirpath+'not valide')

  # get suffix
  aFile = ''
  for filename in os.listdir(outcomedirpath):
    if 'outcome' in filename:
      aFile = filename
      break
  
  suffix = aFile.split('-k-')[1]

  # get params
  paramfile = outcomedirpath+'outcome-k-'+suffix
  print 'Retrive training outcome: ',paramfile
  if not os.path.exists(paramfile):
    sys.exit('Missing Outcomefile '+paramfile)

  value_k,testFeatures = getTrainingOutcome(paramfile)
  params = {'k':value_k,'feat':testFeatures}
  print 'Testing with params: ','k',value_k,'features',testFeatures
  approach = postfix
  run_sunny_oasc(src_path,scenario_test,scenario_train,params,approach)

def main(args):
  if len(args) == 0:
    sys.exit('Missing Arg, E.g. python result.py [scenario] [approach]')

  scenario_name = args[0]
  
  if not len(args) > 1: 
    sys.exit('Missing Arg, E.g. python result.py [scenario] [approach]')

  postfix = args[1]

  validate_trained_scenario(scenario_name,postfix)


if __name__ == '__main__':
  main(sys.argv[1:])
  