SUNNY-AS for OASC challenge
===

This programm is an extension of the SUNNY-AS tool, available at [CP-UNIBO](https://github.com/CP-Unibo/sunny-as).


We integrated a training phase for SUNNY,
and we proposed two modalities: autok and fkvar. 
"autok" studies only the neighbourhood value k, "fkvar" accounts for both k and features.
For more details please refer to [description](https://github.com/lteu/oasc/blob/master/description/main.pdf).


Requirements
============

+ Python v2.x
  https://www.python.org/


Instructions
============
The source codes of SUNNY and training are contained in the folder 'src' and 'oasc' respectively.
The folder 'main' contains the scripts to run different modalities of SUNNY on all scenarios.



The program runs training and testing in sequence, in following we take 
'autok' approach as execution example:

1. Training:

Go to the 'main' folder.
```
 main:$~ sh make_oasc_tasks.sh > tasks.txt 
 main:$~ sh oasc_train.sh run_autok tasks.txt # configured for parallel execution

```

Training results would be stored in the corresponding folder and will be read automatically
by test scripts

2. Testing:
```
 main:$~ sh make_oasc_tasks.sh > tasks.txt 
 main:$~ sh oasc_test.sh autok tasks.txt # configured for parallel execution
```


For the other approach 'fkvar', it is sufficient to substitute 'autok' for 'fkvar', thus,
the following commands:

```
 main:$~ sh make_oasc_tasks.sh > tasks.txt 
 main:$~ sh oasc_train.sh run_fkvar tasks.txt # configured for parallel execution
 main:$~ sh oasc_test.sh run_fkvar tasks.txt # configured for parallel execution
```

Note that,
- The two training modalities should run separately. 
- In 'oasc' folder, you can also run SUNNY on a single scenario as follow (e.g. scenario 'Caren'):
```
 oasc:$~ python run_autok.py Caren # training,
 oasc:$~ python result.py Caren autok # testing
```

Authors
======
- Tong Liu (t.liu at cs.unibo.it)
- Roberto Amadini (roberto.amadini at unimelb.edu.au)
- Jacopo Mauro (mauro.jacopo at gmail.com)


License :copyright:
===
The SUNNY-OASC is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

See http://www.gnu.org/licenses/gpl.html.