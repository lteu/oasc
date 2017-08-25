SUNNY-AS for OASC challenge
===

This programm is an extension of the SUNNY-AS tool, available at [CP-UNIBO](https://github.com/CP-Unibo/sunny-as)


We integrated a training phase for SUNNY algorithm,
and we proposed two approches: autok and fkvar.

"autok" studies only the value k, "fkvar" accounts both k and features.

# Run Scripts

The program runs training and testing in sequence, in following we take 
'autok' approach as execution example:

- Training:

Go to the 'main' folder.
```
 $ sh make_oasc_tasks.sh > tasks1.txt 
 $ sh oasc_train.sh run_autok tasks1.txt # configured for parallel execution

```

Training results would be stored in the corresponding folder and will be read automatically
by testing scripts

- Testing:
```
 $ sh make_oasc_tasks.sh > tasks1.txt 
 $ sh oasc_test.sh autok tasks1.txt # configured for parallel execution
```


For the other approach 'fkvar', it is sufficient to substitute 'autok' for 'fkvar', thus,
the following commands:

```
 $ sh make_oasc_tasks.sh > tasks2.txt 
 $ sh oasc_train.sh run_fkvar tasks2.txt # configured for parallel execution
 $ sh oasc_test.sh run_fkvar tasks2.txt # configured for parallel execution
```

Note that, two training approaches should run separately. 

License
===
The SUNNY-OASC is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License and the above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

See http://www.gnu.org/licenses/gpl.html.