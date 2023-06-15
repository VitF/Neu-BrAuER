import numpy as np
import onnxruntime as ort
import torch


torch2numpy_dataset = False # if True, data and labels are loaded for stored .pt file, converted to numpy array and saved as .npy

whole_set = True # if False, random single-sampole inference is run from the test set
Ns = 100 # number of single-sample inferences, used only if whole_set is False


net = "snnTorch_HAR_statequant_12b8jdeq_ju8w6.onnx" #"snnTorch_HAR_statequant_12b8jdeq_ju8w6-ALTERNATIVE.onnx" #"snnTorch_HAR_dummy.onnx"


if torch2numpy_dataset:

    dataset_filename = "./data/dataset_splits/watch_subset2_40/watch_subset2_40_ds_test.pt" #'watch_subset2_40_ds_test.pt'

    dataset = torch.load(dataset_filename)
    input_tensors = []
    input_labels = []
    for i in dataset:
        input_tensors.append(i[0].numpy())
        input_labels.append(i[1].numpy())

    data = np.array(input_tensors)
    labels = np.array(input_labels)

    np.save('./data/dataset_splits/watch_subset2_40/numpy_data_watch_subset2_40_ds_test', data)
    np.save('./data/dataset_splits/watch_subset2_40/numpy_labels_watch_subset2_40_ds_test', labels)

else:

    data = np.load('./data/dataset_splits/watch_subset2_40/numpy_data_watch_subset2_40_ds_test.npy')
    labels = np.load('./data/dataset_splits/watch_subset2_40/numpy_labels_watch_subset2_40_ds_test.npy')


providers =  ['CPUExecutionProvider']
session = ort.InferenceSession(net, providers=providers)

outname = [i.name for i in session.get_outputs()]
inname = [i.name for i in session.get_inputs()]


if whole_set:

    preds = []
    for num,el in enumerate(data):
        spk_out = session.run(outname, {inname[0]:np.expand_dims(el,1)})
        # Prediction
        #act_total_out = np.sum(outp[0], 0)  # sum over time
        #neuron_max_act_total_out = np.argmax(act_total_out)  # argmax over output units to compare to labels
        #preds.append(neuron_max_act_total_out)
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
        print("\tSample: {} \t Prediction: {}".format(labels[rnd_idx],pred))
    
    print("------------------------------------------")
    print("Overall accuracy from the {} runs: {}%".format(Ns,np.round(np.mean(check_preds)*100,2)))
    print("------------------------------------------")
