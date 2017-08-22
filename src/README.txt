EXAMPLE (aslib-v2.0)
===================

This is  an example on QBF-2011 scenario of aslib-v2.0, for examples with older aslib version pls refer to [CP-UNIBO](https://github.com/CP-Unibo/sunny-as)

0. SPLIT

   sunny-as:$~ python src/split_scenario.py data/aslib-v2.0/QBF-2011

1. TRAIN

   sunny-as:$~ python src/train_scenario.py data/aslib-v2.0/QBF-2011

2. PRESOLVING [optional]

   sunny-as:$~ python src/pre_process.py --kb-path data/aslib-v2.0/QBF-2011/kb_QBF-2011 
   -E "weka.attributeSelection.InfoGainAttributeEval" -S "weka.attributeSelection.Ranker -N 5" 
   --static-schedule --filter-portfolio data/aslib-v2.0/QBF-2011
    
3. TEST

   sunny-as:$~ python src/test_scenario.py -K data/aslib-v2.0/QBF-2011/kb_QBF-2011 data/aslib-v2.0/QBF-2011

*. For a complete example

   sunny-as:$~ python src/evaluate_scenarios.py