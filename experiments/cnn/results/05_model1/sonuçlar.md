
Epoch [01/15] Train Loss: 0.9978 | Train Acc: 0.6851 || Val Loss: 0.5670 | Val Acc: 0.8233
Epoch [02/15] Train Loss: 0.5094 | Train Acc: 0.8386 || Val Loss: 0.4011 | Val Acc: 0.8689
Epoch [03/15] Train Loss: 0.3621 | Train Acc: 0.8873 || Val Loss: 0.3092 | Val Acc: 0.9004
Epoch [04/15] Train Loss: 0.2573 | Train Acc: 0.9167 || Val Loss: 0.2328 | Val Acc: 0.9268
Epoch [05/15] Train Loss: 0.2236 | Train Acc: 0.9287 || Val Loss: 0.2177 | Val Acc: 0.9331
Epoch [06/15] Train Loss: 0.1678 | Train Acc: 0.9444 || Val Loss: 0.2037 | Val Acc: 0.9403
Epoch [07/15] Train Loss: 0.1490 | Train Acc: 0.9502 || Val Loss: 0.2541 | Val Acc: 0.9222
Epoch [08/15] Train Loss: 0.1293 | Train Acc: 0.9573 || Val Loss: 0.2175 | Val Acc: 0.9342
Epoch [09/15] Train Loss: 0.1214 | Train Acc: 0.9598 || Val Loss: 0.2117 | Val Acc: 0.9387
Epoch [10/15] Train Loss: 0.1004 | Train Acc: 0.9670 || Val Loss: 0.2502 | Val Acc: 0.9309
Epoch [11/15] Train Loss: 0.0843 | Train Acc: 0.9716 || Val Loss: 0.2627 | Val Acc: 0.9314
Epoch [12/15] Train Loss: 0.0890 | Train Acc: 0.9704 || Val Loss: 0.2287 | Val Acc: 0.9441
Epoch [13/15] Train Loss: 0.0790 | Train Acc: 0.9741 || Val Loss: 0.2281 | Val Acc: 0.9464
Epoch [14/15] Train Loss: 0.0677 | Train Acc: 0.9777 || Val Loss: 0.2165 | Val Acc: 0.9466
Epoch [15/15] Train Loss: 0.0669 | Train Acc: 0.9781 || Val Loss: 0.2191 | Val Acc: 0.9519





Test Accuracy: 0.9526
==============================

Confusion Matrix:

[[ 339    5   16    2    0    7    8    5    1    0    1    0    1   20]
 [   8  214    0    0    0    0    1    0    0    0    0    0    0    3]
 [   4    0  247    0    0    1    1    4    1    0    1    0    0    6]
 [   0    0    0  474    0    0    0    0    0    0    0    2    1    0]
 [   2    0    2    0  497    0    1    1    2    1    4    0    2    9]
 [   1    0    0    0    0  822    0    0    0    0    0    0    0    4]
 [   4    0    2    2    0    0  366    4    0    0    0    0    0   11]
 [   0    1    1    0    0    1    5  332    4    2    6    0    0    4]
 [   8    0    2    0    1    0    0    3  265    2    0    0    0   12]
 [   0    0    0    0    0    0    0    0    0   52    0    0    0    5]
 [   3    0    2    0    0    0    0    3    1    0  749    0    0    6]
 [   0    0    2    2    0    0    0    0    0    0    0  270    0    2]
 [   2    0    1    5    0    0    0    0    0    0    0    3  210    2]
 [  24    1    4    5    1    2   12    8   23    0    3    2    1 1760]]





==============================
 Logit Margin Analysis
==============================
Mean margin (correct): 0.9767
Mean margin (wrong)  : 0.6546
Correct margin 10% percentile: 0.9841
Correct margin 25% percentile: 0.9998
Correct margin 50% percentile: 1.0000



==============================
Class-wise Recall Analysis
==============================
Apple           : 0.8370
Blueberry       : 0.9469
Cherry          : 0.9321
Corn            : 0.9937
Grape           : 0.9539
Orange          : 0.9940
Peach           : 0.9409
Pepper,_bell    : 0.9326
Potato          : 0.9044
Raspberry       : 0.9123
Soybean         : 0.9804
Squash          : 0.9783
Strawberry      : 0.9417
Tomato          : 0.9534

------------------------------
Macro Recall: 0.9430
Weighted Recall: 0.9526