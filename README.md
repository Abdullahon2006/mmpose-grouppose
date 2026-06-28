#This project aimed to port transformer Grouppose into MMLab's MMpose engine, to enable training and testing via MMLab's native tools. 
This approach was inspired by adoption of EDPose model.
The main jobs carried out include conversion of checkpoints from grouppose to MMlab compatible naming. Building workflow of the model via configs. And lastly, cuda compatibility configuration for later RTX models(on which it was tested on for coco val 2017).


## Results

**COCO val2017 · ResNet-50 · single-scale · no flip**

| Implementation | AP | AP₅₀ | AP₇₅ | AP_M | AP_L |
|---|---|---|---|---|---|
| Original GroupPose | **72.0** | 88.2 | 78.9 | 66.8 | 80.4 |
| **This port (MMPose)** | **71.9** | 89.5 | 78.9 | 66.7 | 79.7 |

The 0.1 AP gap is within rounding/resize margin — both use identical preprocessing.
