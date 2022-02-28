import copy
from datetime import datetime
import json
import math
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import os
import random
from tqdm import tqdm


class Gesture:  # Gesture object class with attributes for learner and teacher

    def __init__(self, symbol, cd, strength):
        self.symbol = symbol

        self.cd = cd  # gesture's TB upper surface constriction degree
        self.cd_list = []  # gesture's constriction degree series (logged throughout training)

        self.strength = strength  # gesture's TB upper surface alpha (strength) value
        self.strength_list = []  # gesture's strength series (logged throughout training)

    def update_cd(self, rate):  # update gesture's constriction degree value during training
        self.cd = np.clip(round(self.cd + rate, 2), -2, None)

    def log_cd(self):  # log gesture's constriction degree values throughout training
        self.cd_list.append(round(self.cd, 2))

    def update_strength(self, rate):  # update gesture's strength value during training
        self.strength = np.clip(round(self.strength+rate, 2), 1, None)

    def log_strength(self):  # log gesture's strength values throughout training
        self.strength_list.append(round(self.strength, 2))


class Grammar:  # Grammar object class, attribute of Agent class containing constraints

    def __init__(self, phon_inv, con_path='', load_dict=None, progen=False):
        if con_path:  # if creating new grammar from .txt file
            with open(con_path, 'r') as con_table:
                constraint_names = []
                init_weights = []
                teacher_weights = []
                constraint_defs = []
                constraint_def_strs = []

                con_table.readline()  # ignore first line
                con_table.readline()  # ignore second line
                for line in con_table:  # process all other lines as grammar constraints
                    constraint_names.append(line.split('\t')[0])
                    init_weights.append(float(line.split('\t')[1]))
                    teacher_weights.append(float(line.split('\t')[2]))
                    constraint_defs.append(eval(line.split('\t')[3]))
                    if line.split('\t')[3][-1] == '\n':
                        constraint_def_strs.append(line.split('\t')[3][:-1])
                    else:
                        constraint_def_strs.append(line.split('\t')[3])

            self.constraint_names = constraint_names
            self.constraint_defs = constraint_defs
            self.constraint_def_strs = constraint_def_strs

            if progen:  # if grammar of first-generation teacher
                self.weights = np.asarray(teacher_weights)
                self.progen = True
                self.teacher = True
            else:
                self.weights = np.asarray(init_weights)
                self.progen = False
                self.teacher = False
            self.weight_list = np.empty((0, len(self.weights)), float)

        elif load_dict:  # if loading pre-existing grammar from .json file
            self.constraint_names = load_dict['constraint_names']

            self.constraint_defs = []
            for con_def in load_dict['constraint_def_strs']:
                self.constraint_defs.append(eval(con_def))

            self.constraint_def_strs = load_dict['constraint_def_strs']

            self.weights = np.asarray(load_dict['weights'])
            self.progen = load_dict['progen']
            self.teacher = load_dict['teacher']

            self.weight_list = np.asarray(load_dict['weight_list'])

        # Make Lookup Tableau #

        # Make all phonological inputs

        inputs = []
        for v1 in phon_inv:
            for v2 in phon_inv:
                inputs.append(f'{v1}_{v2}')

        # Make all output candidates

        candidates = []
        for v1 in phon_inv:
            for persist in ['+', '']:
                for v2 in phon_inv:
                    for inhibit in ['x', '']:
                        candidates.append(f'{v1}{persist}_{v2}{inhibit}')

        # Make tableau dictionary

        tableau = {}
        for phon_input in inputs:
            tableau[phon_input] = {}
            for candidate in candidates:
                if phon_input.split('_')[0][0] == candidate.split('_')[0][0] and phon_input.split('_')[-1][0] == \
                        candidate.split('_')[-1][0]:
                    tableau[phon_input][candidate] = np.array([])
                    for constraint in self.constraint_defs:
                        tableau[phon_input][candidate] = np.append(tableau[phon_input][candidate],
                                                                   constraint(candidate))

        self.tableau = tableau  # lookup table of violation profiles for all inputs and output candidates

    def eval(self, phon_input, except_cand=None, eval_type='maxent'):  # evaluate output candidates in HG/MaxEnt
        if except_cand is None:
            except_cand = []
        tableau = copy.deepcopy(self.tableau[phon_input])  # dictionary with candidates as keys

        if except_cand:  # remove unwanted candidates during RIPP
            for cand in except_cand:
                tableau.pop(cand)

        if self.progen:  # if progenitor grammar
            eval_type = 'hg'  # evaluate candidates via HG

        if eval_type == 'hg':
            max_harmony = -math.inf
            for cand in tableau.keys():
                harmony = sum(tableau[cand]*self.weights)
                if harmony > max_harmony:
                    max_harmony = harmony
                    phon_output = cand
                    prob_dict = None
        elif eval_type == 'maxent':
            prob_dict = {}
            exp_harmony = np.array([])

            for cand in tableau.keys():
                exp_harmony = np.append(exp_harmony, math.exp(sum(tableau[cand] * self.weights)))
            probs = exp_harmony / sum(exp_harmony)

            for i, cand in enumerate(tableau):
                prob_dict[cand] = probs[i]

            phon_output = random.choices(list(tableau.keys()), weights=probs, k=1)[0]
        else:
            print('Error: Select eval_type "maxent" or "hg" only.')
            return

        return phon_output, prob_dict

    def update_weights(self, con_lr, ewc):  # update constraint weights during training
        self.weights = np.clip(self.weights + con_lr * ewc, 0, None)

    def log_weights(self):  # log constraint weights throughout training
        self.weight_list = np.append(self.weight_list, [self.weights], axis=0)


class Agent:  # learner, teacher, or progenitor (first-generation teacher)

    def __init__(self, load='', grammar_con_path='', gest_param_path='', progen=False):

        if load:  # if loading pre-existing Agent object
            print(f'Loading {load}.')
            with open(load) as jsonfile:
                agent_dict = json.load(jsonfile)  # import agent as dictionary from .json file

            self.filename = agent_dict['filename']
            self.pattern_name = agent_dict['pattern_name']

            self.phon_inv = {}
            for vowel in agent_dict['phon_inv'].values():
                self.phon_inv[vowel['symbol']] = json2gest(vowel)

            self.grammar = Grammar(phon_inv=self.phon_inv, load_dict=agent_dict['grammar'])

            self.teacher = agent_dict['teacher']

        elif grammar_con_path:  # if initializing new Agent

            self.filename = ''  # initialize empty filename for .json file

            with open(grammar_con_path, 'r') as file:
                self.pattern_name = file.readline()[14:-1]

            # Initialize phoneme inventory from .txt file

            self.phon_inv = {}

            with open(gest_param_path, 'r') as phon_inv:
                self.pattern_name += f'_{phon_inv.readline()[14:-1]}'  # first line added to pattern_name
                phon_inv.readline()  # ignore second line
                for line in phon_inv:
                    if progen:
                        self.phon_inv[line.split('\t')[0]] = Gesture(symbol=line.split('\t')[0],
                                                                     cd=float(line.split('\t')[1]),
                                                                     strength=float(line.split('\t')[2]))
                    else:
                        self.phon_inv[line.split('\t')[0]] = Gesture(symbol=line.split('\t')[0],
                                                                     cd=16,
                                                                     strength=random.randint(1, 20))

            # Initialize grammar from constraint set file

            self.grammar = Grammar(phon_inv=self.phon_inv, con_path=grammar_con_path, progen=progen)

            # Initialize as learner/progenitor

            self.teacher = progen

        else:
            print('Enter either an agent filename to load a saved agent OR grammar and inventory filenames to train a '
                  'new agent.')

    def mature(self):  # learner-to-teacher maturation in generational learner
        self.teacher = True

    def produce(self, phon_output):  # produce phonological output according to current gestural parameter settings
        v1, v2 = phon_output.split('_')  # examples: ['i+', 'a'] ['e', 'ax']
        v1_height = self.phon_inv[v1[0]].cd
        if v1[-1] == '+' and v2[-1] != 'x':  # if v1 persistent and v2 non-blocking
            v2_height = blend(self.phon_inv[v1[0]], self.phon_inv[v2[0]])
        else:
            v2_height = self.phon_inv[v2[0]].cd

        return np.array([v1_height, v2_height])

    def train(self, teacher_agent, con_lr, gest_lr, window, n_iter=0):  # train learner's grammar and gest params
        if self.teacher:  # prevent training of a teacher agent
            print('Cannot train a teacher agent.')
            return
        elif not teacher_agent.teacher:  # prevent learning from a non-teacher agent
            print('Cannot learn from a non-teacher agent.')
            return
        else:  # training for learner agents only
            progress = tqdm()  # initialize progress bar

            if n_iter == 0:  # if training to convergence instead of specified number of iterations
                n_iter = 200000  # then change number of iterations to 200000

            for i in range(n_iter):
                # create training item
                phon_input = random.choice(list(self.grammar.tableau.keys()))  # Randomly select input (uniform dist)

                learner_phon_output, _ = self.grammar.eval(phon_input)  # learner selects/samples output according to its grammar
                learner_production = self.produce(learner_phon_output)  # teacher selects/samples output according to its grammar

                teacher_phon_output, _ = teacher_agent.grammar.eval(phon_input)  # learner produces output according to its gest params
                teacher_production = teacher_agent.produce(teacher_phon_output)  # teacher produces output according to its gest params

                # Check for production error

                if max(abs(teacher_production - learner_production)) >= window:  # if production error

                    # do RIPP (Robust Interpretive Production Parsing)

                    candidates = self.grammar.tableau[phon_input].keys()  # get candidates

                    cand_prod_diffs = {cand_key: sum(abs(teacher_production - self.produce(cand_key))) for cand_key in candidates}  # dict of candidates vs teacher production differences
                    min_cand_production_diff = min(cand_prod_diffs.values())
                    non_min_outputs = [key for key, value in cand_prod_diffs.items() if
                                       value != min_cand_production_diff]

                    teacher_rip_output, _ = self.grammar.eval(phon_input, except_cand=non_min_outputs)

                    # Do learner-teacher grammar comparison and updates

                    if learner_phon_output != teacher_rip_output:
                        ewc = self.grammar.tableau[phon_input][teacher_rip_output] - self.grammar.tableau[phon_input][learner_phon_output]
                        self.grammar.update_weights(con_lr, ewc)  # update constraint weights

                    # Do learner gestural parameter updates

                    learner_rip_production = self.produce(teacher_rip_output)  # learner produces teacher RIP output with its current gestural parameters
                    ht_diffs = teacher_production - learner_rip_production  # difference in height between teacher and learner RIPs

                    active = []  # initialize list of active gestures

                    for output_v in range(len(teacher_rip_output.split('_'))):  # iterate through each vowel slot
                        v_ht_diff = ht_diffs[output_v]  # get vowel height difference vector

                        # determine which vowels are active during this V slot

                        if "x" in teacher_rip_output.split('_')[output_v]:  # if current vowel is blocker...
                            active = []  # ...then deactivate all previous gestures
                        active.append((teacher_rip_output.split('_')[output_v],
                                       self.phon_inv[teacher_rip_output.split('_')[output_v][0]]))  # activate current V

                        # perform vowel updates

                        if abs(v_ht_diff) >= window:  # if vowel has production error
                            if len(active) > 1:  # if there are any overlapping persistent vowels
                                cd_difference = active[1][1].cd - active[0][1].cd  # determine which vowel is higher

                                active[1][1].update_cd(math.copysign(gest_lr, v_ht_diff))  # update current vowel cd
                                active[1][1].update_strength(math.copysign(gest_lr, v_ht_diff * cd_difference))  # update current vowel strength

                                active[0][1].update_cd(math.copysign(gest_lr, v_ht_diff))  # update overlapping vowel(s) cd
                                active[0][1].update_strength(math.copysign(gest_lr, v_ht_diff*-cd_difference))  # update overlapping vowel(s) strength

                            else:  # if there is no overlap/blending
                                active[0][1].update_cd(math.copysign(gest_lr, v_ht_diff))  # update current vowel cd
                            if "+" not in teacher_rip_output.split('_')[output_v]:  # if current vowel is not persistent...
                                _ = active.pop()  # ...then remove it from active

                # Log constraint weights

                self.grammar.log_weights()  # log constraint weights

                # Log updated gestural parameter values

                for v in self.phon_inv.values():  # for each vowel being trained...
                    v.log_strength()  # ...log its strength after this trial
                    v.log_cd()  # ...and log its constriction degree after this trial

                progress.update(1)

                # Check for convergence if training to convergence

                if n_iter == 200000:  # if n_iter set to 200,000 due to training to convergence
                    if self.__check_convergence(teacher_agent=teacher_agent, window=window):
                        break

    def __check_convergence(self, teacher_agent, window):
        for phon_input in self.grammar.tableau:
            prob_correct = 0  # initialize counter for probability of correct production

            teacher_production = teacher_agent.produce(teacher_agent.grammar.eval(phon_input, eval_type='hg')[0])
            _, learner_candidates = self.grammar.eval(phon_input)
            for candidate in learner_candidates:
                learner_production = self.produce(candidate)
                if max(abs(teacher_production - learner_production)) < window:  # if produced correctly
                    prob_correct += learner_candidates[candidate]  # add candidate probability to counter

            if prob_correct < 0.99:  # if bank is less than 99%
                return False  # return False (not converged)

        return True  # return True (converged) if all inputs produced correctly ≥99% of time

    def report_training(self, prnt=True, export=False):  # print/return report of agent's state after training
        report = '\n# Learner Training Report #\n'
        report += '\nConstraint Weights\n\n'
        for x in range(len(self.grammar.constraint_names)):
            report += f'{self.grammar.constraint_names[x]}: {self.grammar.weights[x]}\n'
        report += '\nGestural Parameters\n\n'
        for phon in self.phon_inv.values():
            report += f'/{phon.symbol}/: CD = {phon.cd}, Strength = {phon.strength}\n'
        report += '\nOutputs Generated\n\n'
        for phon_input in self.grammar.tableau:
            for candidate in self.grammar.tableau[phon_input]:
                report += f'/{phon_input}/ -> [{candidate}], ' \
                          f'[{self.produce(candidate)[0]}, ' \
                          f'{self.produce(candidate)[1]}], ' \
                          f'{self.grammar.eval(phon_input)[1][candidate]}\n'
            report += '\n'
        report += f'Training Iterations Conducted: {len(self.grammar.weight_list)}\n'

        if prnt:
            print(report)
        if export:
            return report

    def plot_training(self, axis=''):  # visualize changes in agent's constraint weights and gest params over time

        colors = pl.cm.viridis(np.linspace(0, 1, len(self.phon_inv)))  # make a colormap for plotting

        plt.figure('Gestural Strength Learning Trajectories')

        x = 0  # initialize counter for colormap
        ymax = 0  # initialize maximum y value for plot

        for v in self.phon_inv.values():
            if max(v.strength_list) > ymax:
                ymax = max(v.strength_list)  # record new maximum y value
            plt.plot(v.strength_list, '-', color=colors[x], label=v.symbol)  # plot vowel strength
            x += 1  # iterate counter for colormap
            length = len(v.strength_list)
        plt.axis([0, length, 0, plt_round(ymax, 10)])
        if axis == 'log':
            plt.yscale('log')
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.legend(loc=(1.04, 0.5))

        plt.figure('Constriction Degree Learning Trajectories')

        x = 0  # re-initialize counter for colormap

        for v in self.phon_inv.values():
            plt.plot(v.cd_list, '-', color=colors[x], label=v.symbol)  # plot vowel constriction degree
            x += 1
        plt.axis([0, length, -5, 20])
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        plt.legend(loc=(1.04, 0.5))

        plt.figure('Constraint Weight Learning Trajectories')

        colors = pl.cm.viridis(np.linspace(0, 1, len(self.grammar.weights)))  # make a new colormap for plotting

        x = 0  # re-initialize counter for colormap
        ymax = 0  # re-initialize maximum y value for plot

        for w in range(len(self.grammar.weights)):
            if max(self.grammar.weight_list[:, w]) > ymax:
                ymax = max(self.grammar.weight_list[:, w])
            plt.plot(self.grammar.weight_list[:, w], '-', color=colors[x], label=self.grammar.constraint_names[w])
            x += 1
            length = len(self.grammar.weight_list[:, w])
        plt.axis([0, length, 0, plt_round(ymax, 5)])
        plt.tight_layout(rect=[0, 0, 0.7, 1])
        plt.legend(loc=(1.04, 0.5))

        plt.show()

    def dict(self):  # make Agent object json serializable for saving

        agent_dict = copy.deepcopy(self.__dict__)  # copy object dictionary

        for vowel in agent_dict['phon_inv']:
            agent_dict['phon_inv'][vowel] = agent_dict['phon_inv'][vowel].__dict__  # serialize vowels in phon inventory

        agent_dict['grammar'] = agent_dict['grammar'].__dict__  # serialize agent grammar
        agent_dict['grammar']['weights'] = agent_dict['grammar']['weights'].tolist()  # weights from numpy array to list
        agent_dict['grammar']['weight_list'] = agent_dict['grammar']['weight_list'].tolist() # weight_list from numpy array to list

        _ = agent_dict['grammar'].pop('constraint_defs')  # remove constraint definition lambdas from grammar dict
        _ = agent_dict['grammar'].pop('tableau')  # remove tableau from grammar dict

        return agent_dict

    def save(self, directory='', check_sure=True):  # save agent as .json file

        if directory:  # if directory specified
            if not os.path.isdir(directory):  # if directory does not exist
                print('Directory not found. Try again.')
                return
            else:  # if directory does exist
                directory += '/'

        if self.filename and check_sure and input(f'Agent file {self.filename} already exists. Do you want to replace '
                                                  f'it? (y / n)') in ['y', 'Y', 'yes', 'Yes', 'YES']:  # check if sure
            filename = self.filename
        else:
            it = 1  # initialize model marker

            while os.path.isfile(f'{directory}{self.pattern_name}_{it}.json'):  # check to see if saved file exists
                it += 1  # if so, iterate its marker until you find one not already in use
            filename = f'{self.pattern_name}_{it}.json'  # append marker to filename
            print(f'Saving agent as {filename}.')

            self.filename = filename  # record the model's filename

        with open(f'{directory}{filename}', 'w') as model_json_file:
            json.dump(self.dict(), model_json_file)  # save to current directory
            model_json_file.close()

####################
# HELPER FUNCTIONS #
####################


def blend(*gests):
    numerator = 0
    denominator = 0
    for gest in gests:
        numerator += gest.cd * gest.strength
        denominator += gest.strength

    return numerator / denominator


def json2gest(json_dict):  # parse a .json dictionary into a Gesture object

    gesture = Gesture(symbol=json_dict['symbol'],
                      cd=json_dict['cd'],
                      strength=json_dict['strength'])

    gesture.cd_list = json_dict['cd_list']
    gesture.strength_list = json_dict['strength_list']

    return gesture


def plt_round(x, rnd):
    if x % rnd == 0:
        return x + 10
    else:
        mult = rnd ** -1
        return math.ceil(x*mult) / mult


#############################################
# BATCH AND GENERATIONAL TRAINING FUNCTIONS #
#############################################

def train_generations(model_type, n_gens, n_iter, specs=(0.1, 0.1, 0.2)):
    if model_type == 'chainshift':
        con_path = './con_hightrigger3.txt'
        gest_path = './gestparams_chainshift3.txt'
    elif model_type == 'saltationblend':
        con_path = './con_hightrigger3.txt'
        gest_path = './gestparams_saltation3.txt'
    elif model_type == 'saltationblock':
        con_path = './con_hightrigger_midblock3.txt'
        gest_path = './gestparams_saltation3.txt'
    else:
        print('Try again. Choose model_type "chainshift", "saltationblend", or "saltationblock".')
        return

    con_lr, gest_lr, window = specs

    start_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    os.mkdir(f'gentraining_{start_time}')

    with open(f'./gentraining_{start_time}/specs.txt', 'w') as spec_file:
        spec_file.write(f'Model Type: {model_type}\n')
        spec_file.write(f'Constraint Learning Rate: {con_lr}\n')
        spec_file.write(f'Gestural Parameter Learning Rate: {gest_lr}\n')
        spec_file.write(f'Window: {window}')
        spec_file.close()

    teacher = Agent(grammar_con_path=con_path, gest_param_path=gest_path, progen=True)

    for gen in range(n_gens):
        learner = Agent(grammar_con_path=con_path, gest_param_path=gest_path)
        learner.train(teacher_agent=teacher, con_lr=con_lr, gest_lr=gest_lr, window=window, n_iter=n_iter)
        learner.save(f'gentraining_{start_time}')

        learner.mature()
        teacher = learner

    with open(f'./gentraining_{start_time}/lastgen_report.txt', 'w') as report_file:
        report_file.write(learner.report_training(export=True))
        report_file.close()


def train_batches(model_type, n_models, specs=(0.1, 0.1, 0.2)):
    if model_type == 'chainshift':
        con_path = './con_hightrigger3.txt'
        gest_path = './gestparams_chainshift3.txt'
    elif model_type == 'saltationblend':
        con_path = './con_hightrigger3.txt'
        gest_path = './gestparams_saltation3.txt'
    elif model_type == 'saltationblock':
        con_path = './con_hightrigger_midblock3.txt'
        gest_path = './gestparams_saltation3.txt'
    else:
        print('Try again. Choose model_type "chainshift", "saltationblend", or "saltationblock".')
        return

    con_lr, gest_lr, window = specs

    start_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    os.mkdir(f'trainingbatch_{start_time}')

    with open(f'./trainingbatch_{start_time}/specs.txt', 'w') as spec_file:
        spec_file.write(f'Model Type: {model_type}\n')
        spec_file.write(f'Constraint Learning Rate: {con_lr}\n')
        spec_file.write(f'Gestural Parameter Learning Rate: {gest_lr}\n')
        spec_file.write(f'Window: {window}')
        spec_file.close()

    teacher = Agent(grammar_con_path=con_path, gest_param_path=gest_path, progen=True)

    for _ in range(n_models):
        learner = Agent(grammar_con_path=con_path, gest_param_path=gest_path)
        learner.train(teacher_agent=teacher, con_lr=con_lr, gest_lr=gest_lr, window=window)
        learner.save(f'trainingbatch_{start_time}')

        with open(f'./trainingbatch_{start_time}/convergence_times.txt', 'a') as conv_file:
            conv_file.write(f'{str(len(learner.grammar.weight_list))}\n')
