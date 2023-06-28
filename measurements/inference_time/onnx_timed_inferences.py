import numpy as np
import onnxruntime as ort
from playsound import playsound
from time import sleep
import time
from repeated_timer import RepeatedTimer
import sys

# You can pass the number of minutes as an argument
# If no argument is passed, 10 is used as default.


def stop_inferences():
    global perform_inference 
    perform_inference = False


minutes = 10
n_inferences = 0
perform_inference = True
net = "snnTorch_Braille_statequant_x6k9dqcj_YKa5r.onnx" #"snnTorch_HAR_statequant_12b8jdeq_ju8w6-ALTERNATIVE.onnx" #"snnTorch_HAR_dummy.onnx"
device = "cpu"

if len(sys.argv) == 2:
    minutes = int(sys.argv[1]) 

f = open("times.txt", "wt")
f.write("**********************\n")
f.write("Execution for minutes: "+ str(minutes) + "\n")
f.write("**********************\n\n")

# Load Data
timestart = time.time()	
print("Loading data...")
data = np.load('./data/numpy_data_braille_letters_digits_40Hz_augmented_ds_test.npy')
print("Data loaded in " + str(time.time() - timestart) + " s")    

# Load Labels
timestart = time.time()
print("Loading labels...")
labels = np.load('./data/numpy_labels_braille_letters_digits_40Hz_augmented_ds_test.npy')
print("Labels loaded in " + str(time.time() - timestart) + " s")

# Load Model
timestart = time.time()
print("Loading model...")

letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

providers =  ['CPUExecutionProvider']
session = ort.InferenceSession(net, providers=providers)

print("Model loaded in " + str(time.time()-timestart) + " s")


outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]

rt = RepeatedTimer(minutes*60, stop_inferences)
print("Will run inferences for " + str(minutes) + " minute(s)")

while perform_inference:
    rnd_idx = np.random.randint(data.shape[0])
    timestart = time.time()
    spk_out = session.run(outname, {inname[0]:np.expand_dims(data[rnd_idx],1)})
    delta = time.time() - timestart
    n_inferences = n_inferences + 1
    f.write(str(delta) + "\n")        

rt.stop()
print("Stop. Total inferences: " + str(n_inferences))

f.close()
