import time

from psychopy import core, visual, event, data
from psychopy.hardware import keyboard
from psychopy.preferences import prefs
from port_open_send import *
from sklearn.utils import shuffle
import numpy as np

import sklearn
import pandas as pd
import random
import os
import re

ROOT = '/Users/meguser/Desktop/Experiments/Stefan/MEG_2/' #'/Users/amilleahrodriguez/' #'/Users/alr664/Desktop' #

stimuli_select = input("which experiment (practice, experiment): ")
participant_name = input("participant: ")

simulation_time = time.time()
debug = False
toggle_diode = True
send_triggers = True
simulate_responses = False
num_blocks = 7

# exp_info = {'Participant:':"test0",
# 			'Exp_type': ['practice', 'experiment'],
# 			'Response_simulation': True,
# 			'Debug_mode':False,
# 			'toggle_diode': True,
# 			'send_triggers': True,
# 			'fullscreen?': False}
#
# dlg = gui.DlgFromDict(dictionary=exp_info, title='Experiment settings')
# if dlg.OK == False: # quit if press cancel
#     core.quit()
#
#
# simulation_time = time.time()
# participant_name = exp_info["Participant:"]
# simulate_responses = exp_info["Response_simulation"]
# debug = exp_info["Debug_mode"]
# toggle_diode = exp_info["toggle_diode"]
# send_triggers = exp_info["send_triggers"]

if simulate_responses: print("Reponse Simulation...")

# Pick the stimuli you will use (practice or experiment)
exp_type = stimuli_select #exp_info['Exp_type']
if exp_type == "practice":
	stimuli_fn = 'practice_stims.csv'
	practice = True
elif exp_type == 'experiment':
	stimuli_fn = 'target_stims.csv'


dir_logs = os.path.join(ROOT, 'logs')
stimuli_path = os.path.join(ROOT, stimuli_fn)

log_files = [os.path.join(dir_logs, f) for f in os.listdir(dir_logs) if re.match(r'R\d{4}.+', f)]
num_logs = len(log_files)

# if num_logs % 2 == 0:

print("reading in data from:", stimuli_path)
print("current num_logs: ", num_logs)

if debug: print("DEBUG IS ON")
if not toggle_diode: print("DIODE IS OFF")
if not send_triggers: print("TRIGGERS IS SET TO OFF")
if simulate_responses: print("SIMULATED RESPONSES")

setup_text = 'Tenga paciencia mientras preparamos el experimento. Por favor, no presione ningún botón.'
instruction_text = "Va a ver varias palabras en Español. Pulse la tecla usando el dedo índice izquierdo si ve una palabra que existe en Español. Pulse la tecla usando el dedo medio si ve una palabra que no es una palabra que existe. \n\nPulse qualquier tecla cuando esté listo para comenzar"
end_text = 'Ha completado el experimento. Por favor, no se mueva mientras tomamos las últimas medidas.'

setup_proceed_key = ['s']
instruction_proceed_keys = ['s', '1', '2']
end_proceed_key = ['s', 'q']

"""================================Objects=================================="""
# define monitor.

# create window, stimuli, and photodiode objects
win = visual.Window(monitor='default', units='pix', fullscr=True, colorSpace='rgb255', color=(127,127,127))
sentence = visual.TextStim(win, text='', font='Courier New', height = 20, wrapWidth=700, alignText='center', color=(1,1,1))
fixation = visual.TextStim(win, text='+', color=(1, 1, 1))
intertrialblank = visual.TextStim(win, text='',color=(127,127,127))
taskText = visual.TextStim(win, text='', font='Courier New', wrapWidth=700, alignText='center', color=(1, 1, 1))
endText = visual.TextStim(win=win, text='', font='Courier New', pos=(0, 0), height=18, wrapWidth=700, ori=0.0, color='white', colorSpace='rgb', opacity=1, languageStyle='LTR', depth=0.0);
photodiode = visual.Rect(win, width=57, height=57, pos=[-483,355], fillColor=[255,255,255], fillColorSpace='rgb255') #-483,355
instructions = visual.TextStim(win,text='', font='Courier New', wrapWidth=700, alignText='center', color=(1,1,1))

proceed = keyboard.Keyboard()
response = keyboard.Keyboard()
trialClock = core.Clock()
clock = core.Clock()

# For checking if user presses escape
defaultKeyboard = keyboard.Keyboard()

win.mouseVisible = False # hides the mouse

# send inital trigger to reset channels
win.callOnFlip(sendTrigger, channel='ch161', duration=0.02)
delay = 600
proceed = keyboard.Keyboard()

"""================================Timings=================================="""
frame_time_ms = win.monitorFramePeriod * 1000

fixation_ON_ms = 300
fixation_OFF_ms = 300

sent_ON_ms = 300
sent_OFF_ms = 300

"""==============================Get Stimuli================================"""

trials = pd.read_csv(stimuli_path)
trials = shuffle(trials)

for index, row in trials.iterrows():

	trials.at[index, 'target'] = row['target']
	trials.at[index, 'correct_resp'] = row['correct_ans']
	trials.at[index, 'trigger'] = row['trigger']
	trials.at[index, 'target_type'] = row['target_type']

trials['correct_resp'] = trials['correct_resp'].astype(int)
trials['trigger'] = trials['trigger'].astype(int)


# Assumes that experiment --> number of trials will be divisible by 7
if exp_type == 'practice':
	n_blocks = 1
elif exp_type == 'experiment':
	n_blocks = num_blocks

# Get the indices of the rows in the DataFrame
trial_indices = trials.index.values

# Shuffle the indices
np.random.shuffle(trial_indices)

# Determine the number of trials per block
n_trials_per_block = len(trial_indices) // n_blocks
remainder_trials = len(trial_indices) % n_blocks

# Create blocks to store DataFrames for each block
blocks = []

for i, block in enumerate(range(n_blocks)):

	start_idx = i * n_trials_per_block

	# assign all remainder trials to final block
	if i == n_blocks - 1:
		end_idx = (i + 1) * n_trials_per_block + remainder_trials
	else:
		end_idx = ((i + 1) * n_trials_per_block)

	# Extract indices for the current block
	block_indices = trial_indices[start_idx:end_idx]

	# Get the DataFrame slice for the current block using the indices
	block_df = trials.iloc[block_indices]

	# Append the DataFrame slice to blocks
	blocks.append(block_df)


"""==========================Display Functions============================="""


def please_wait(text='',img='', proceed_keys= ['s']):
	instructions.setText(text)
	instructions.draw()

	win.flip()
	return event.waitKeys(keyList=proceed_keys)

def present_text(text='',img='', proceed_keys= ['s','1','2','q']):
	instructions.setText(text)
	instructions.draw()

	win.flip()
	return event.waitKeys(keyList=proceed_keys)

def present_fixation(fixation_time=fixation_ON_ms, fixation_off_time=fixation_OFF_ms,frame_time=frame_time_ms):

	win.callOnFlip(trialClock.reset)
	fixation.draw()

	t0 = time.time()

	total_onscreen_time = int(fixation_time/frame_time)
	print(total_onscreen_time)

	for i in range(total_onscreen_time):

		fixation.draw()

		win.flip()

	stim_time = time.time()
	# print('fixation onscreen time:', str(stim_time - t0))

	for i in range(int(fixation_off_time/frame_time)):
		win.flip()

	t1 = time.time()


# Pause slide is entered and exited by pressing 'p'
def pause_slide():
	print("ETNER")
	sentence.setText("Please wait a bit...")
	sentence.draw()
	win.flip()

	response = event.waitKeys(keyList=['p'])

	if response[0] == 'p':
		return()
	elif response[0] == 'q':
		core.quit()

def trial_slide(trial, send_triggers=send_triggers, toggle_photodiode=toggle_diode, sent_onscreen_ms=sent_ON_ms, sent_offscreen_ms=sent_OFF_ms, frame_time=frame_time_ms):
	print('Stimulus:', str(trial['target']).upper())
	countdown_timer = core.CountdownTimer(0)
	win.callOnFlip(sendTrigger, channel='ch165', duration=0.02)

	target = trial['target']
	print(target)

	sentence.setText(target)
	sentence.draw()


	if toggle_photodiode:
		photodiode.draw()

	win.flip()
	rt0 = time.time()

	#If the slide is paused, the "response" for that trial will be "p"
	if event.getKeys('p'):
		pause_slide()
		return ("p", time.time() - rt0)

	if simulate_responses:
		if event.getKeys('q'):
			core.quit()
		response = str(random.randint(1,2))
		core.wait(random.uniform(0.5,0.9))
		return (response, time.time() - rt0,)

	else:
		response = event.waitKeys(keyList=['1', '2', 'q', 's'])



		if response[0] == 'q':
			core.quit()

		return (response[0], time.time() - rt0)

def pause_experiment():
    present_text("experiment paused. \n please wait for the experimenter.")

"""============================Experiment=================================="""


t0 = time.time()

please_wait(setup_text, setup_proceed_key)
present_text(instruction_text, instruction_proceed_keys)

if not debug or participant_name != "test":
	exp = data.ExperimentHandler(dataFileName=dir_logs+'/%s_logfile'%(participant_name), autoLog=False, savePickle=False)

trialNum = 0
blockNum = 0
trial_counter = 0

for blockNum, block in enumerate(blocks, start=1):

	if blockNum != 1:
		accuracy = "Ha acertado %d%% de las veces." % (int((total_correct/total_answered)*100))
		prompt = '\n\n Este es el bloque %s de %s.\n\n Recuerde moverse lo menos posible e intente parpadear lo menos posible cuando avance al siguiente bloque. \n\n Presione cualquier botón para continuar.' % (
		    str(blockNum), len(blocks))
		full_text = accuracy + prompt
		present_text(full_text)
	else:
		prompt = 'Este es el bloque %s de %s.\n\n Recuerde moverse lo menos posible e intente parpadear lo menos posible cuando avance al siguiente bloque. \n\n Presione cualquier botón para continuar.' % (
		    str(blockNum), len(blocks))
		present_text(prompt)

	total_correct = 0
	total_answered = 0

	for trialNum, trial in block.iterrows():  # Change here to use iterrows() instead of enumerate

		trial_counter += 1

		print(trial)
		present_fixation()

		response, rt = trial_slide(trial, trialNum)

		if exp_type == "practice":
			if int(response) == trial['correct_resp']:
				sentence.setText("¡Correcto!")
				sentence.draw()

				win.flip()
				core.wait(1)
			elif int(response) == 1 and trial['correct_resp'] == 2:
				sentence.setText("¡Incorrecto!" + str(" ¡" + trial['target'] + " es una palabra real!"))
				sentence.draw()

				win.flip()
				core.wait(2)

			elif int(response) == 2 and trial['correct_resp'] == 1:
				sentence.setText("¡Incorrecto! ¡" + str(trial['target'] + " no es una palabra real!"))
				sentence.draw()

				win.flip()
				core.wait(2)

		# present_sentence(trial, trialNum)
		#
		# response, rt = present_answer(trial, trialNum)
		total_answered += 1

		print("response:" + str(response))
		print(type(response))
		print("correct response: " + str(trial['correct_resp']))
		print(type(trial['correct_resp']))

		print(response == trial['correct_resp'])


		if response == str(trial['correct_resp']):
		    total_correct += 1

		# core.wait(random.uniform(0.6, 0.8))
		accuracy = "%d/%d" % (total_correct, total_answered)

		correct = True if response == str(trial['correct_resp']) else False

		win.flip()

		exp.addData('participant', participant_name)
		exp.addData('stimuli', trial.target)
		exp.addData('trialNum', trial_counter)
		exp.addData('target_type', trial['target_type'])
		exp.addData('response', response)
		exp.addData('correct', correct)
		exp.addData('block', blockNum)
		exp.addData('reaction_time', rt)
		exp.addData('trigger', trial['trigger'])

		exp.nextEntry()

		core.wait(random.uniform(0.6, 0.8))

# ========== End Experiment ==========
simulation_time = time.time() - simulation_time
print("simulation run time = " + str(simulation_time))

wait_text = end_text
present_text(wait_text, proceed_keys = ['q'])
win.close()

t1 = time.time()
print("Total Experiment Time:", str(t1 - t0))
