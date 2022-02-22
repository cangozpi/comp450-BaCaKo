###  Reinforcement learning approach to algorithmic trading (_BaCaKo_)

---
* This project investigates the effect of introducing intrinsic rewards to policy gradient methods in the field of algorithmic trading.
* Aim of this project is to apply reinforcement learning to train agents to trade lucratively in the
stock market.
* This project was done collaboratively as part of the *Reinforcement Learning (Comp450, Koc University)* group term project requirement.

---
To train a model change to type variable in line 14 of main to select the type of model then run main.
  1. 0 to train a short term model
  2. 1 to train a long term model
  3. 2 to train a short term model with intrinsic reward mechanism
  4. 3 to train a long term model with itnrinsic reward mechanism
*Note that a previously trained long term model is required to be in the lt directory to train a short term intrinsic reward model.
A previously trained short term model is required to be in the st directory to train a long term intrinsic reward model.*

* To validate a model, upload the model checkpoint file to lt, st, lt_ir, st_ir. 
Lt stands for long term, st for short term and _ir for intrinsic reward versions.
In the validation pyhton notebook cell 2 the network class, change the path to the corresponding model. 
There are comments under the line which should be changed.
Also change the type to 0 or 1 in the 4th cell which is the main. 
0 for short term and short term with intrinsic.
1 for long term and long term with intrinsic.

For further information please refer to the *FinalReport* pdf.
