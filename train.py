import os
import tensorflow as tf

from mediapipe_model_maker import gesture_recognizer

import matplotlib.pyplot as plt

# List available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("Available GPUs:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPUs available")

labels = []
for dir_name in os.listdir('HandTyper_MAIN/data'):
    if os.path.isdir('HandTyper_MAIN/data/' + dir_name):
        labels.append(dir_name)

print(labels)

NUM_EXAMPLES = 50
dataset_path = 'HandTyper_MAIN/data/'

for label in labels:
  label_dir = os.path.join(dataset_path, label)
  example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
  fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))

  for i in range(NUM_EXAMPLES):
    print(os.path.join(label_dir, example_filenames[i]))
    axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
  fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')
  
  plt.close(fig)


print(labels)
dataset = gesture_recognizer.Dataset.from_folder(
  dirname=dataset_path,
  hparams=gesture_recognizer.HandDataPreprocessingParams()
  )
train, rest = dataset.split(0.8)
validation, test = rest.split(0.5)

hparams = gesture_recognizer.HParams(export_dir="ONE_HAND_v2")
options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)

model = gesture_recognizer.GestureRecognizer.create(
  train,
  validation,
  options=options
  )

loss, acc = model.evaluate(test, batch_size=5)
print(f'Loss: {loss}, Accuracy: {acc}')
model.export_model()