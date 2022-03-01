## The Grammar + Gesture Gradual Learning Algorithm

This project introduces the Grammar + Gesture Gradual Learning Algorithm (GGGLA), an error-driven online learning algorithm, and applies it to the task of learning derivationally opaque height harmony patterns. The algorithm is introduced in the following talk:

Smith, Caitlin (2022) Grammar and Representation Learning for Opaque Harmony Processes (joint work with [Charlie O'Hara](https://charlieohara.github.io/)). Invited talk presented at the Society for Computation in Linguistics, Online, February 2022.

The code we use for computational modeling of the learning of height harmony can be found in `gggla.py` above. Here, we will walk you through how to use this code.

**Check python version and install dependencies.** The code for the GGGLA learner is written for use with python 3.8. It is likely compatible with some older versions of python 3, but we make no guarantees. The code uses several packages that are not included in the python standard library and must be installed by the user. These are: `matplotlib` (pip install), `numpy` (conda/pip install), and `tqdm` (pip install).

**Create new agents from a harmony pattern file.** A pattern file should be a .txt file specifying a vowel inventory. Several sample .txt files are included in the repository. Use the `Agent` class to initialize new agent objects, both learners and teachers.

```
>>> learner = Agent(grammar_con_path='./con_hightrigger_noblock3.txt', gest_param_path='./gestparams_chainshift3.txt')
>>> teacher = Agent(grammar_con_path='./con_hightrigger_noblock3.txt', gest_param_path='./gestparams_chainshift3.txt', progen=True)
```
**Train the learner agent.** Once the agent is initialized, train the height gestures of its vowels using the `train()` method. This can take anywhere from a few seconds to a minute.

`>>> learner.train(teacher_agent=teacher, con_lr=0.1, gest_lr=0.1, window=0.2)`

Output:

`11384it [00:04, 2832.59it/s]`

If the learner's constraint weights and gestural parameter settings do not converge to values that reproduce the teacher's data after 200k iterations, training ceases. If that occurs, you can rerun `train()`, which will pick up training where it left off.

Output:

`200000it [05:42, 583.42it/s]`

**Inspect the learner's results.** For a text display of the final states of the learner's vowel inventory and constraint weights, use the `report_training()` method.

`>>> learner.report_training()`

Output:
```
# Learner Training Report #

Constraint Weights

Persist(height): 0.10000000000000003
SelfDeactivate: 4.899999999999999
*Gest(TBhigh,SelfDeact): 10.19999999999998
*Inhibit: 5.1999999999999975
Inhibit(TBheight,TBheight): 0.0
Inhibit(TBhigh,TBmid): 0.0
Inhibit(TBhigh,TBlow): 0.0
Inhibit(TBmid,TBlow): 2.800000000000001

Gestural Parameters

/i/: CD = 3.9, Strength = 24.2
/e/: CD = 9.8, Strength = 1.4
/a/: CD = 16.2, Strength = 22.4

Outputs Generated

/i_i/ -> [i+_ix], [3.9, 3.9], 0.005461631000955918
/i_i/ -> [i+_i], [3.9, 3.9], 0.9900420958381029
/i_i/ -> [i_ix], [3.9, 3.9], 2.4667898494500524e-05
/i_i/ -> [i_i], [3.9, 3.9], 0.004471605262446766

/i_e/ -> [i+_ex], [3.9, 9.8], 0.005461631000955918
/i_e/ -> [i+_e], [3.9, 4.22265625], 0.9900420958381029
/i_e/ -> [i_ex], [3.9, 9.8], 2.4667898494500524e-05
/i_e/ -> [i_e], [3.9, 9.8], 0.004471605262446766

/i_a/ -> [i+_ax], [3.9, 16.2], 0.005461631000955918
/i_a/ -> [i+_a], [3.9, 9.81244635193133], 0.9900420958381029
/i_a/ -> [i_ax], [3.9, 16.2], 2.4667898494500524e-05
/i_a/ -> [i_a], [3.9, 16.2], 0.004471605262446766

/e_i/ -> [e+_ix], [9.8, 3.9], 4.478230513426693e-05
/e_i/ -> [e+_i], [9.8, 4.22265625], 0.00811778884802564
/e_i/ -> [e_ix], [9.8, 3.9], 0.005441516594316153
/e_i/ -> [e_i], [9.8, 3.9], 0.9863959122525239

/e_e/ -> [e+_ex], [9.8, 9.8], 4.478230513426693e-05
/e_e/ -> [e+_e], [9.8, 9.8], 0.00811778884802564
/e_e/ -> [e_ex], [9.8, 9.8], 0.005441516594316153
/e_e/ -> [e_e], [9.8, 9.8], 0.9863959122525239

/e_a/ -> [e+_ax], [9.8, 16.2], 0.0006789030531318171
/e_a/ -> [e+_a], [9.8, 15.823529411764707], 0.007483668100028094
/e_a/ -> [e_ax], [9.8, 16.2], 0.08249379344079086
/e_a/ -> [e_a], [9.8, 16.2], 0.9093436354060492

/a_i/ -> [a+_ix], [16.2, 3.9], 4.478230513426693e-05
/a_i/ -> [a+_i], [16.2, 9.81244635193133], 0.00811778884802564
/a_i/ -> [a_ix], [16.2, 3.9], 0.005441516594316153
/a_i/ -> [a_i], [16.2, 3.9], 0.9863959122525239

/a_e/ -> [a+_ex], [16.2, 9.8], 0.0006789030531318171
/a_e/ -> [a+_e], [16.2, 15.823529411764707], 0.007483668100028094
/a_e/ -> [a_ex], [16.2, 9.8], 0.08249379344079086
/a_e/ -> [a_e], [16.2, 9.8], 0.9093436354060492

/a_a/ -> [a+_ax], [16.2, 16.2], 4.478230513426693e-05
/a_a/ -> [a+_a], [16.2, 16.2], 0.00811778884802564
/a_a/ -> [a_ax], [16.2, 16.2], 0.005441516594316153
/a_a/ -> [a_a], [16.2, 16.2], 0.9863959122525239

Training Iterations Conducted: 11384
```

To see the trajectories of the learning of each vowel's constriction degree target and blending strength, and each constraint's weight, use the `plot_training()` method.

`>>> model_language.plot_training()`

![](https://caitlinsmith14.github.io/resource/trajectories.png)

**Save the model.** To save the model for inspection at another time, use the `save()` method. This creates a human-readable .json file containing a python dictionary with all of the model's parameters and results.

`>>> learner.save()`

Output:

`Saving agent as hightrigger_noblock3_chainshift3_1.json.`

**Load a pretrained model.** To load a previously trained and saved agent, use the `Agent` class to initialize a new agent object and provide it with a .json file.

`>>> learner2 = Agent(load='hightrigger_noblock3_chainshift3_1.json')`