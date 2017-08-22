#! /usr/bin/env python

'''
Clearn training Results

Author: hearntest
'''

import os
import sys
from subprocess import Popen

#=======================================================
#=======================================================
#=======================================================
def main(args):
  if len(args) == 0:
    sys.exit('Missing Arg, eg: python clean.py fkvar')

  approach_name = args[0]

  scenarios = [
  "Bado",
  "Camilla",
  "Caren",
  "Magnus",
  "Mira",
  "Monty",
  "Oberon",
  "Quill",
  "Sora",
  "Svea",
  "Titus"
  ]

  root_arr = os.path.realpath(__file__).split('/')[:-2]
  root = '/'.join(root_arr) 

  for scenario_name in scenarios:

    scenario_path = root + '/data/oasc_scenarios/train/'+scenario_name
    training_result = scenario_path + '/outcome-'+approach_name
    cmd = 'rm -rf ' + training_result
    print cmd
    proc = Popen(cmd.split())
    proc.communicate()


if __name__ == '__main__':
  main(sys.argv[1:])
  