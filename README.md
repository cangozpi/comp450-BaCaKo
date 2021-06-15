# comp450-BaCaKo
comp450 group project

To train a model change to type variable in line 14 of main to select the type of model then run main.
0 to train a short term model
1 to train a long term model
2 to train a short term model with intrinsic reward mechanism
3 to train a long term model with itnrinsic reward mechanism
Note that a previously trained long term model is required to be in the lt directory to train a short term intrinsic reward model.
A previously trained short term model is required to be in the st directory to train a long term intrinsic reward model.

To validate a model, upload the model checkpoint file to lt, st, lt_ir, st_ir. 
Lt stands for long term, st for short term and _ir for intrinsic reward versions.
In the validation pyhton notebook cell 2 the network class, change the path to the corresponding model. 
There are comments under the line which should be changed.
Also change the type to 0 or 1 in the 4th cell which is the main. 
0 for short term and short term with intrinsic.
1 for long term and long term with intrinsic.


Batu Helvacıoğlu
Can Gözpınar
Koray Tecimer
