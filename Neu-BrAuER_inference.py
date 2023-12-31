#-----*******************************************************-----#
# 
# Author:
#   Vittorio Fra (VitF), Politecnico di Torino, Italy
# Last modified: 2023.10.12
#
#-----*******************************************************-----#


import numpy as np
import onnx
import onnxruntime as ort
from playsound import playsound
from time import sleep


torch2numpy_dataset = True # if True, data and labels are loaded from stored .pt file and converted to numpy array
torch2numpy_save = False # if True (with torch2numpy_dataset=True), data and labels are loaded from stored .pt file, converted to numpy array and saved as .npy

whole_set = True # if False, random single-sampole inference is run from the test set
Ns = 100 # number of single-sample inferences, used only if whole_set is False

model_version = 1.0 # defined by the model version on Zenodo

net = "./models/Neu-BrAuER_v{}.onnx".format(model_version)

if onnx.checker.check_model(net, full_check=True) == None:
    print("#######################")
    print("ONNX model check: done!")
    print("#######################")

device = "cpu"

if torch2numpy_dataset:

    import torch

    dataset_filename = "./data/braille_letters_digits_40Hz_augmented_ds_test.pt"

    dataset = torch.load(dataset_filename, map_location=device)

    input_tensors = []
    input_labels = []
    for i in dataset:
        input_tensors.append(i[0].numpy())
        input_labels.append(i[1].numpy())
    
    data = np.array(input_tensors)
    labels = np.array(input_labels)

    if torch2numpy_save:
        np.save('./data/numpy_data_braille_letters_digits_40Hz_augmented_ds_test', data)
        np.save('./data/numpy_labels_braille_letters_digits_40Hz_augmented_ds_test', labels)

else:

    data = np.load('./data/numpy_data_braille_letters_digits_40Hz_augmented_ds_test.npy')
    labels = np.load('./data/numpy_labels_braille_letters_digits_40Hz_augmented_ds_test.npy')

letter_written = ['Space', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


providers =  ['CPUExecutionProvider']
session = ort.InferenceSession(net, providers=providers)

outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]


if whole_set:

    preds = []
    for num,el in enumerate(data):
        spk_out = session.run(outname, {inname[0]:np.expand_dims(el,1)})
        preds.append( np.argmax( np.sum(spk_out[0], 0) ) )

    print("-----------------")
    print("Accuracy: {}%".format( np.round( np.mean( (np.array(preds)==labels) )*100,2 )) )
    print("-----------------")

else:

    check_preds = np.zeros(Ns)

    for ii in range(Ns):

        rnd_idx = np.random.randint(data.shape[0])

        spk_out = session.run(outname, {inname[0]:np.expand_dims(data[rnd_idx],1)})

        pred = np.argmax(np.sum(spk_out[0],0))

        if pred == labels[rnd_idx]:
            check_preds[ii] = 1

        print("Run {}/{}:".format(ii+1,Ns))
        print("\tSample: {} \t Prediction: {}".format(letter_written[labels[rnd_idx]],letter_written[pred]))

        playsound("./data/character_playback/{}.wav".format(letter_written[pred]))

        sleep(1)
    
    print("------------------------------------------")
    print("Overall accuracy from the {} runs: {}%".format(Ns,np.round(np.mean(check_preds)*100,2)))
    print("------------------------------------------")
