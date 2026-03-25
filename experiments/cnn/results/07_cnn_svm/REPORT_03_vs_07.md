# Global CNN (03) vs CNN+SVM (07)

## Run Paths

- CNN03: `/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/03_all_dataset`

- SVM07: `/Users/duhanaydin/cursor/plant disease study/experiments/cnn/results/07_cnn_svm`

## Overall Metrics (eval_summary.json)

| Metric | CNN03 | SVM07 |
|---|---:|---:|
| test_accuracy | 0.9230 | 0.9408 |
| macro_recall | 0.8851 | 0.9107 |
| weighted_recall | 0.9230 | 0.9408 |
| macro_f1 | 0.8940 | 0.9179 |
| weighted_f1 | 0.9227 | 0.9407 |

## Lowest Recall Classes in CNN03 (top-15)

| class | recall_cnn03 | support | recall_svm07 | delta_recall |
| --- | --- | --- | --- | --- |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.525641 | 78 | 0.730769 | 0.20512799999999998 |
| Potato___healthy | 0.625 | 24 | 0.625 | 0.0 |
| Tomato___Early_blight | 0.693333 | 150 | 0.713333 | 0.020000000000000018 |
| Apple___Apple_scab | 0.789474 | 95 | 0.810526 | 0.02105199999999996 |
| Tomato___Septoria_leaf_spot | 0.801498 | 267 | 0.850187 | 0.04868899999999998 |
| Apple___Black_rot | 0.808511 | 94 | 0.882979 | 0.07446799999999998 |
| Apple___Cedar_apple_rust | 0.809524 | 42 | 0.809524 | 0.0 |
| Potato___Late_blight | 0.813333 | 150 | 0.88 | 0.06666700000000003 |
| Tomato___Target_Spot | 0.816038 | 212 | 0.872642 | 0.05660399999999999 |
| Tomato___Late_blight | 0.860627 | 287 | 0.850174 | -0.010453000000000046 |
| Corn___Northern_Leaf_Blight | 0.872483 | 149 | 0.825503 | -0.04698000000000002 |
| Peach___healthy | 0.872727 | 55 | 0.890909 | 0.01818199999999992 |
| Grape___Esca_(Black_Measles) | 0.879808 | 208 | 0.947115 | 0.067307 |
| Raspberry___healthy | 0.894737 | 57 | 0.964912 | 0.07017499999999999 |
| Tomato___Tomato_mosaic_virus | 0.894737 | 57 | 0.894737 | 0.0 |

## Biggest Recall Gains (SVM07 - CNN03) (top-15)

| class | recall_cnn03 | support | recall_svm07 | delta_recall |
| --- | --- | --- | --- | --- |
| Corn___Cercospora_leaf_spot Gray_leaf_spot | 0.525641 | 78 | 0.730769 | 0.20512799999999998 |
| Apple___Black_rot | 0.808511 | 94 | 0.882979 | 0.07446799999999998 |
| Raspberry___healthy | 0.894737 | 57 | 0.964912 | 0.07017499999999999 |
| Grape___Esca_(Black_Measles) | 0.879808 | 208 | 0.947115 | 0.067307 |
| Potato___Late_blight | 0.813333 | 150 | 0.88 | 0.06666700000000003 |
| Tomato___Target_Spot | 0.816038 | 212 | 0.872642 | 0.05660399999999999 |
| Tomato___Septoria_leaf_spot | 0.801498 | 267 | 0.850187 | 0.04868899999999998 |
| Cherry___Powdery_mildew | 0.937107 | 159 | 0.981132 | 0.04402499999999998 |
| Grape___Black_rot | 0.915254 | 177 | 0.949153 | 0.03389900000000001 |
| Grape___healthy | 0.90625 | 64 | 0.9375 | 0.03125 |
| Strawberry___Leaf_scorch | 0.904192 | 167 | 0.934132 | 0.029939999999999967 |
| Strawberry___healthy | 0.942029 | 69 | 0.971014 | 0.02898500000000004 |
| Peach___Bacterial_spot | 0.939306 | 346 | 0.962428 | 0.023121999999999976 |
| Apple___Apple_scab | 0.789474 | 95 | 0.810526 | 0.02105199999999996 |
| Tomato___Leaf_Mold | 0.902778 | 144 | 0.923611 | 0.02083299999999999 |

## Biggest Recall Drops (SVM07 - CNN03) (top-15)

| class | recall_cnn03 | support | recall_svm07 | delta_recall |
| --- | --- | --- | --- | --- |
| Corn___Northern_Leaf_Blight | 0.872483 | 149 | 0.825503 | -0.04698000000000002 |
| Tomato___Late_blight | 0.860627 | 287 | 0.850174 | -0.010453000000000046 |
| Orange___Haunglongbing_(Citrus_greening) | 1.0 | 827 | 1.0 | 0.0 |
| Potato___healthy | 0.625 | 24 | 0.625 | 0.0 |
| Corn___healthy | 0.994286 | 175 | 0.994286 | 0.0 |
| Apple___Cedar_apple_rust | 0.809524 | 42 | 0.809524 | 0.0 |
| Tomato___Tomato_mosaic_virus | 0.894737 | 57 | 0.894737 | 0.0 |
| Tomato___Tomato_Yellow_Leaf_Curl_Virus | 0.978882 | 805 | 0.980124 | 0.0012419999999999654 |
| Tomato___Spider_mites Two-spotted_spider_mite | 0.90873 | 252 | 0.912698 | 0.0039679999999999715 |
| Tomato___healthy | 0.983333 | 240 | 0.9875 | 0.004167000000000032 |
| Blueberry___healthy | 0.977876 | 226 | 0.982301 | 0.004425000000000012 |
| Potato___Early_blight | 0.953333 | 150 | 0.96 | 0.0066669999999999785 |
| Cherry___healthy | 0.945736 | 129 | 0.953488 | 0.007751999999999981 |
| Soybean___healthy | 0.971204 | 764 | 0.980366 | 0.009162000000000003 |
| Corn___Common_rust | 0.972222 | 180 | 0.983333 | 0.011110999999999982 |

## Margin Comparison

Important: CNN03 margin = **logit top1-top2**; SVM07 margin = **decision-score top1-top2**. Scales differ; compare separation (correct vs wrong), not absolute values.

### margin_stats.json

| Source | mean_correct | mean_wrong | median_correct | median_wrong | n_correct | n_wrong |
|---|---:|---:|---:|---:|---:|---:|
| CNN03 (json) | 6.9466 | 1.0895 | 5.8930 | 0.7152 | 7549 | 630 |
| SVM07 (json) | 1.0935 | 1.0159 | 1.0041 | 1.0011 | 7695 | 484 |

### per_sample_margin.csv (computed)

| Source | mean_correct | mean_wrong | median_correct | median_wrong | n_correct | n_wrong |
|---|---:|---:|---:|---:|---:|---:|
| CNN03 (csv) | 6.9466 | 1.0895 | 5.8930 | 0.7152 | 7549 | 630 |
| SVM07 (csv) | 1.0935 | 1.0159 | 1.0041 | 1.0011 | 7695 | 484 |
