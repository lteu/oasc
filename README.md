OASC challenge
===

We integrated a training phase for SUNNY algorithm,
and we proposed two approches: autok and fkvar.

"autok" studies only the value k, "fkvar" accounts both k and features.

# Run Scripts

The program runs training and testing in sequence, in following we take 
'autok' approach as execution example:

- Training:

Go to the 'main' folder.
···
 $ sh make_oasc_tasks.sh > tasks1.txt 
 $ sh oasc_train.sh run_autok tasks1.txt # configured for parallel execution

···

Training results would be stored in the corresponding folder and will be read automatically
by testing scripts

- Testing:
···
 $ sh make_oasc_tasks.sh > tasks1.txt 
 $ sh oasc_test.sh autok tasks1.txt # configured for parallel execution
···


For the other approach 'fkvar', it is sufficient to substitute 'autok' for 'fkvar', thus,
the following commands:

···
 $ sh make_oasc_tasks.sh > tasks2.txt 
 $ sh oasc_train.sh run_fkvar tasks2.txt # configured for parallel execution
 $ sh oasc_test.sh run_fkvar tasks2.txt # configured for parallel execution
···

Note that, two training approaches should run separately.