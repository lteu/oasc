Execution Note:

Training:


go to main folder,

 $ sh make_oasc_tasks.sh > tasks.txt
*$ sh oasc_train.sh run_autok tasks.txt 
*$ sh oasc_train.sh run_fkvar tasks.txt 

Test:

*$ sh oasc_test.sh autok tasks.txt


Note:

For the other approach please substitute "run_autok" with "run_fkvar"
the scripts are configured for parallel execution, so that, the * parts can be run in parallel with more machines. However, two training approaches should run sequentially.



todo:

running new fkvar on Sora, Titus

run results.py on fkvar

check results.py of autok i think they won't change much