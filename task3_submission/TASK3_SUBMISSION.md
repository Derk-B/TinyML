# Task 3 Submission

Final model: 64-64-32 MLP

## Validation result

Task: 3  
Split: validation  
Source: clean  
Input features: 36  

Mean positioning error: 1.548 cm  
Median positioning error: 1.172 cm  
P95 positioning error: 4.173 cm  

Mean device invoke time: 14.875 ms  
TFLite model size: 38344 bytes  
Tensor arena: 81920 bytes  
UF2 firmware size: 275968 bytes  
 Final files

Final UF2 firmware:

firmware/build/vlp_pico_task3_64_64_32_1p548cm.uf2

Task 3 model files:

models/task3_64_64_32_1p548cm.tflite
models/task3_model_64_64_32_1p548cm.pt
models/task3_scaling_64_64_32_1p548cm.npz

Reproduce

Train:

python scripts/train_task3.py

Export TFLite and firmware assets:

python scripts/export_litert_task3.py

Build firmware:

./firmware/build_firmware.sh

Evaluate:

python host/run_submission.py \
  --task 3 \
  --split validation \
  --source clean \
  --port /dev/cu.usbmodemXXXX \
  --uf2 firmware/build/vlp_pico.uf2
