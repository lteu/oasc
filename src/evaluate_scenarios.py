'''
Evaluate different ASlib scenarios.

launch with:
sunnay-as$:~ python src/evaluate_scenarios.py

if only for stats
sunnay-as$:~ python src/evaluate_scenarios.py stats
'''

from __future__ import division
import os
import csv
import json
import sys
from subprocess import Popen

from analyze_scenario import analyze_scenario_stats
from analyze_scenario import analyze_log

ASLIB_VERSION = 'aslib-v2.0'


if len(sys.argv) >= 2:
    flag = sys.argv[1]
else:
    flag = 'World'


in_path = os.path.realpath(__file__).split('/')[:-2]
# List of the scenarios to test.

scenarios = [
  # 'ASP-POTASSCO',
  # 'CSP-2010', 
  # 'MAXSAT12-PMS',
  # 'PREMARSHALLING-ASTAR-2015',
  # 'PROTEUS-2014',
  # 'QBF-2011',
  'SAT11-HAND',
  # 'SAT11-INDU',
  # 'SAT11-RAND',
  # 'SAT12-ALL',
  # 'SAT12-HAND',
  # 'SAT12-INDU',
  # 'SAT12-RAND',
]  
for scenario in scenarios:

  

  print 'Evaluating scenario',scenario

  src_path = '/'.join(in_path) + '/src/'
  path = '/'.join(in_path) + '/data/'+ASLIB_VERSION+'/' + scenario
  
  if flag != 'stats':

    print 'Splitting scenario',scenario

    cmd = 'python ' + src_path + 'split_scenario.py ' + path
    proc = Popen(cmd.split())
    proc.communicate()

    # for subdir, dirs, files in os.walk(path + '/cv_' + scenario):
    for subdir in os.listdir(path + '/cv_' + scenario):
    
      if 'train_' in subdir and 'kb_' not in subdir:
        subdir = path + '/cv_' + scenario+"/"+subdir
        
        print 'Training',subdir
        
        options = ' --discard --feat-timeout +inf '
        cmd = 'python ' + src_path + 'train_scenario.py ' + options + subdir
        proc = Popen(cmd.split())
        proc.communicate()
        
        test_dir = subdir.replace('train_', 'test_')
        kb_name = subdir.split('/')[-1]
        pred_file = test_dir + '/predictions.csv'
        

        # print 'Pre-processing',test_dir

        # # options = [
        # #     '--kb-path', subdir + '/kb_' + kb_name,
        # #     '-E', 'weka.attributeSelection.InfoGainAttributeEval',
        # #     #'-E', 'weka.attributeSelection.GainRatioAttributeEval',
        # #     #'-E', 'weka.attributeSelection.SymmetricalUncertAttributeEval',
        # #     #'-E', 'weka.attributeSelection.ReliefFAttributeEval',
        # #     '-S', 'weka.attributeSelection.Ranker -N 5', 
        # #     '--static-schedule', '--filter-portfolio'
        # # ]
        # options = [
        #   '--kb-path', subdir + '/kb_' + kb_name,
        #   # '-E', 'wrapper',
        #   # '-E', 'weka.attributeSelection.GainRatioAttributeEval',
        #   #'-E', 'weka.attributeSelection.SymmetricalUncertAttributeEval',
        #   #'-E', 'weka.attributeSelection.ReliefFAttributeEval',
        #   # '-S', '6,7,12', 
        #   # '-S', 'weka.attributeSelection.Ranker -N 5',
        #   '--static-schedule', '--filter-portfolio'
        # ]
        # options = [
        #   '--kb-path', subdir + '/kb_' + kb_name,
        #   '-E', 'wrapper',
        #   # '-E', 'weka.attributeSelection.GainRatioAttributeEval',
        #   #'-E', 'weka.attributeSelection.SymmetricalUncertAttributeEval',
        #   #'-E', 'weka.attributeSelection.ReliefFAttributeEval',
        #   '-S', '2,9,16,18,19', 
        #   # '-S', 'weka.attributeSelection.Ranker -N 5',
        #   '--static-schedule', '--filter-portfolio'
        # ]
        # cmd = ['python', src_path + 'pre_process.py'] + options + subdir.split()
        # proc = Popen(cmd)
        # proc.communicate()
        
        print 'Testing',test_dir

        options = ' ' 
        # -f container-density,group-same-mean,stacks,group-same-stdev,tiers '
        # InfoGain Selected Features (5 features).
        # ASP: Running_Avg_LBD-4,Learnt_from_Loop-1,Frac_Learnt_from_Loop-1,Literals_in_Conflict_Nogoods-1,Literals_in_Loop_Nogoods-1
        # CSP: stats_Local_Variance,stats_tightness_75,normalised_width_of_graph,normalised_median_degree,stats_cts_per_var_mean
        # MAX-SAT: horn,vcg_var_spread,vcg_var_min,vcg_var_max,vcg_cls_mean
        # PREMARSHALLING: container-density,group-same-mean,stacks,group-same-stdev,tiers
        # PROTEUS: csp_perten_avg_predshape,csp_perten_avg_predsize,csp_sqrt_max_domsize,csp_sqrt_avg_domsize,directorder_reducedVars
        # QBF: FORALL_POS_LITS_PER_CLAUSE,EXIST_VARS_PER_SET,LITN_LIT,OCCP_OCCN,NEG_HORN_CLAUSE
        # SAT11-HAND: BINARYp,horn_clauses_fraction,SP_bias_q25,VCG_CLAUSE_coeff_variation,lobjois_mean_depth_over_vars
        # SAT11-INDU: saps_BestAvgImprovement_Mean,VG_min,VG_coeff_variation,VG_max,CG_coeff_variation
        # SAT11-RAND: VCG_CLAUSE_min,saps_FirstLocalMinStep_Q10,gsat_BestSolution_Mean,cl_size_mean,nclauses
        # SAT12-ALL: SP_unconstraint_q25,vars_clauses_ratio,SP_unconstraint_mean,SP_unconstraint_q75,POSNEG_RATIO_CLAUSE_entropy
        # SAT12-HAND: reducedClauses,SP_bias_coeff_variation,horn_clauses_fraction,SP_unconstraint_max,POSNEG_RATIO_CLAUSE_min
        # SAT12-INDU: POSNEG_RATIO_VAR_entropy,VCG_VAR_coeff_variation,VCG_VAR_entropy,reducedVars,POSNEG_RATIO_VAR_stdev
        # SAT12-RAND: VCG_VAR_mean,VCG_CLAUSE_mean,saps_BestSolution_Mean,VCG_CLAUSE_min,VCG_CLAUSE_max
        
      #   cmd = 'python ' + src_path + 'test_scenario.py ' + options + ' -k '+ str(5)+' -o ' + pred_file \
      # + ' --print-static -K ' + subdir + '/kb_' + kb_name + ' ' + test_dir 
      #   cmd = 'python ' + src_path + 'test_scenario.py ' + options + ' -o ' + pred_file \
      # + ' --print-static -K ' + subdir + '/kb_' + kb_name + ' ' + test_dir 

      #   # print cmd
      #   # sys.exit()
      #   proc = Popen(cmd.split())
      #   proc.communicate()

  # statistics and evaluation
  avgfsi_sunny, avgfsi_vbs, avgpar10_sunny, avgpar10_vbs,m,n,scenario,avgfsi_sbs,avgpar10_sbs = analyze_scenario_stats(path)

  print '\n==========================================='
  print 'Scenario:',scenario
  print 'No. of instances:',n,'(',m,'solvable )'
  print 'FSI SUNNY:', avgfsi_sunny
  print 'FSI VBS:', avgfsi_vbs
  print 'FSI SBS:', avgfsi_sbs
  print 'PAR 10 SUNNY:', avgpar10_sunny
  print 'PAR 10 VBS:', avgpar10_vbs
  print 'PAR 10 SBS:', avgpar10_sbs
  print '---'
  print 'PAR 10 Gap:', round((avgpar10_sbs - avgpar10_sunny)/(avgpar10_sbs - avgpar10_vbs),4)
  print '===========================================\n'

  # analyze_log(path)

# Results with --discard, --static-schedule, -f f1,...,f5
#   PAR10 FSI
#ASP    600.0 0.905
#CSP    6606.0  0.870
#MAXSAT   3420.0  0.840
#PREMARSH   1655.7  0.964
#PROTEUS  5146.0  0.859
#QBF    8980.8  0.756
#SAT11-HAND   18241.4 0.642
#SAT11-INDU   12765.5 0.753
#SAT11-RAND   9921.8  0.805
#SAT12-ALL    1458.2  0.888
#SAT12-HAND   4633.1  0.622
#SAT12-INDU   2854.8  0.770
#SAT12-RAND   3291.9  0.729

# Scenario: QBF-2011
# No. of instances: 1368 ( 1054 solvable )
# FSI SUNNY: 0.75365497076
# FSI VBS: 0.770467836257
# PAR 10 SUNNY: 9064.79063596
# PAR 10 VBS: 8337.09934942

