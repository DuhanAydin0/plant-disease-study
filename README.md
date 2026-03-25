# Plant Disease Classification

## An Experimental Deep Learning Portfolio Project

This repository documents an experimental plant disease classification project built on the PlantVillage dataset.

I did **not** build this project to present it as a production-ready agricultural diagnosis product. I built it as a **serious portfolio project** to show how I approach machine learning problems when the first reasonable-looking solution is still not good enough.

The main purpose of the project was to work through questions like these:

* How far can classical ML baselines go before they hit structural limits?
* What should I do when a global CNN gives a strong overall score but still struggles on specific classes?
* When low-recall classes appear, is the right response better tuning, targeted intervention, or a different problem formulation altogether?
* Can a hierarchical pipeline make the task more interpretable?
* Can a hybrid approach such as **CNN + SVM** improve the weak regions of a pure CNN solution?
* How much does **transfer learning** improve representation quality compared to from-scratch training?

Because of that, this repository should be read as an **iterative ML research-and-engineering portfolio project** rather than a single-model showcase.

---

# 1. Why I built this project this way

A lot of plant disease classification projects stop after training one CNN and reporting a good accuracy score.

I did not want to stop there.

What interested me more was the part that usually gets skipped:
**what happens after the first decent result?**

More specifically, I wanted to understand what to do when:

* the model looks strong globally,
* some classes are still weak locally,
* targeted fixes help one area but damage another,
* and the project starts pushing back against the original formulation itself.

That is the real center of this repository.

The final model matters, but the more valuable part of the project is the reasoning trail behind it:
baseline -> limitation -> targeted fix -> failure analysis -> redesign -> stronger formulation.

---

# 2. Scope and positioning

This project uses the **PlantVillage** dataset for controlled image-based plant disease classification.

I want the scope to be very clear:

* This is **not** a field-deployed crop diagnosis system.
* This is **not** a production-hardened commercial product.
* This is **not** a claim that controlled-dataset performance automatically transfers to real agricultural conditions.

It is an **experimental technical portfolio project** built to demonstrate:

* structured experimentation,
* class-wise analysis,
* evidence-based architecture decisions,
* failure-driven redesign,
* and lightweight inference/deployment integration.

That framing matters. I do not think this project becomes stronger by pretending it is more productized than it really is. Its real strength is that it shows how I reason through model behavior, not just how I report a final number.

---

# 3. Core problem idea

The project started from a simple observation.

A single combined-label classifier such as:

* `Tomato___Early_blight`
* `Potato___Late_blight`
* `Apple___healthy`

can produce acceptable overall results, but it also forces one model to learn:

* plant identity,
* disease identity,
* and cross-plant class boundaries

all at the same time.

That can work up to a point, but it can also hide important weaknesses.

So the project gradually turned into an attempt to answer this question:

**What should I do when a model is globally “good enough,” but still weak in class-specific ways that matter?**

That question shaped almost every major decision in the repository.

---

# 4. Dataset strategy

## Dataset

- **PlantVillage**
- Controlled leaf images
- **38 classes**
- **54,305 total images** used in the split-based global classification setup
- Good for comparative experimentation, but limited in real-world variability
- Reference link: https://www.kaggle.com/datasets/mohitsingh1804/plantvillage

## Split summary

- **Train:** 37,997 images
- **Validation:** 8,129 images
- **Test:** 8,179 images

The dataset was organized into explicit train / validation / test directories, and duplicate checks were also run on the generated splits as part of the dataset verification process.

## Data organization

The repository follows a `raw` / `processed` separation:

- `data/raw/` -> untouched source data
- `data/processed/` -> resized, split, and model-ready data

## Splitting philosophy

The project includes scripts for:

- tomato-specific dataset splitting,
- full dataset splitting,
- and Model-2-specific dataset generation.

That was intentional. I treated dataset preparation as part of the experiment logic, not just preprocessing boilerplate.



# 5. Experiment roadmap

The project evolved through several stages.

## Stage A – Classical ML baselines

* **KNN**
* **SVM**
* optimized SVM variant

**Purpose**

* establish baseline comparisons,
* understand shallow-model limitations,
* justify the move to CNNs with evidence.

## Stage B – CNN from scratch

* tomato-only baseline CNN
* optimized tomato CNN
* full 38-class global CNN

**Purpose**

* move beyond flattened-image representations,
* learn spatial features directly,
* test whether CNNs solve the limitations seen in SVM.

## Stage C – Class-specific interventions

* focus-class fine-tuning
* targeted augmentation
* full-dataset augmentation

**Purpose**

* improve difficult low-recall classes,
* test whether the problem is local or structural,
* see whether intervention helps without damaging global balance.

## Stage D – Hierarchical reformulation

* **Model-1**: plant type classifier
* **Model-2**: plant-specific disease classifier

**Purpose**

* separate plant identification from disease classification,
* reduce confusion inside a large global label space,
* make the pipeline easier to reason about.

## Stage E – Hybrid and advanced methods

* **CNN + SVM**
* **Transfer Learning (ResNet-based)**

**Purpose**

* improve difficult class boundaries,
* compare representation quality,
* test whether stronger pretrained backbones outperform from-scratch learning.

---

# 6. Why the numbered experiment order matters

The numbering inside `experiments/` is intentional.

I did not organize those folders as unrelated trials. I used them as a narrative structure:

1. What can simple models do?
2. What improves when I move to CNNs?
3. What still remains weak even after a strong global CNN?
4. Can low-recall areas be repaired locally?
5. If local repair is unstable, should I rethink the task itself?
6. If I split the problem hierarchically, what improves and what still stays difficult?
7. Can a hybrid decision boundary help where the global CNN struggled?
8. What happens when I introduce pretrained representations?

That sequence is one of the main reasons I think this project is stronger than a typical single-model portfolio repo.

---

# 7. Detailed experiment story

## 7.1 KNN baseline

KNN was my simplest baseline. I used it to see how a distance-based shallow learner behaves on image classification when the input is fundamentally high-dimensional and visual.

It was useful as a starting point, but structurally limited:

* high-dimensional pixel space is weak for raw nearest-neighbor classification,
* there is no learned feature extraction,
* and subtle disease differences are hard to separate meaningfully.

I never expected KNN to become a final solution. Its role was to anchor the lower bound of the project and make later improvements easier to interpret.

---

## 7.2 SVM baseline and optimized SVM

The SVM stage was much more important.

It gave me a stronger non-deep-learning baseline and helped answer a question that I think matters in many junior projects:

**Is the model weak because I have not tuned it enough, or because the representation itself is inadequate?**

From the SVM experiments:

* the basic version reached very high train accuracy,
* validation performance dropped noticeably,
* the optimized version reduced overfitting somewhat,
* minority recall improved in some cases,
* but overall generalization still did not improve enough.

That led me to a conclusion I trusted:

**the main bottleneck was not just regularization or class weighting; it was the limitation of flattened-image features for a visual disease classification task.**

That is why moving to CNNs felt justified. It was not escalation for its own sake.

---

## 7.3 01 – Baseline CNN (tomato-only)

The first CNN stage focused on a narrower tomato disease setup.

This gave me a cleaner environment to test whether a custom CNN from scratch was worth pursuing.

It established several things:

* a custom CNN pipeline was viable,
* spatial feature learning clearly helped compared to shallow baselines,
* but the model still showed overfitting behavior.

Approximate baseline tomato test results:

* **Accuracy:** `0.9115`
* **Macro Recall:** `0.8831`
* **Macro F1:** `0.8778`

The important part was not just the score. It was the pattern:

* the model learned quickly,
* train accuracy became very high,
* validation behavior was less stable,
* and visually similar disease classes still produced confusion.

So the first CNN stage was encouraging, but it also reminded me that “CNN works” is not the same as “the problem is solved.”

---

## 7.4 02 – Optimized CNN (tomato-only)

Instead of immediately switching to a completely different architecture, I first tried to improve the original CNN in a controlled way.

The changes included:

* better regularization,
* batch normalization,
* dropout,
* lower learning rate,
* scheduler usage,
* and class weights.

This step mattered a lot to me because it reflects how I want to approach ML work in general:
first improve the current formulation carefully, then decide whether a bigger redesign is necessary.

This stage helped stabilize the pipeline and made the later expansion to the full dataset much more meaningful.

---

## 7.5 03 – Global 38-class CNN

This is one of the most important stages in the repository.

After the earlier tomato-only work, I expanded the task into a **single 38-class global classifier** over the full dataset.

The model produced strong overall results:

* **Test Accuracy:** `0.92297`
* **Macro Recall:** `0.88510`
* **Macro F1:** `0.89398`

If I had stopped at top-line metrics, this would have looked like a finished success.

I did not want to stop there.

I went deeper into:

* class-wise recall,
* margin behavior,
* weak decision boundaries,
* and confusion among visually similar categories.

That analysis showed that some classes were still problematic even though the model looked strong overall.

Examples identified in the experiment analysis included:

* `Corn___Cercospora_leaf_spot Gray_leaf_spot`
* `Tomato___Early_blight`
* `Potato___healthy`
* some Apple disease categories

This changed the direction of the project.

At that point the question was no longer:
**Can I get decent accuracy?**

It became:
**Why are some classes still weak even when the global model is already strong?**

That question pulled the project into a much more interesting direction.

---

## 7.6 03_focus_classes – focused fine-tuning

After identifying the weak classes, I did not immediately jump to a new architecture.

The next logical step was to test whether the problem was still local:
could I focus training around the difficult classes and improve recall without disturbing the rest of the system too much?

That led to the **focus-class fine-tuning** stage.

### What improved

Some target classes improved substantially:

* `Corn___Cercospora_leaf_spot Gray_leaf_spot`

  * recall: `0.5256 -> 0.8077`
* `Tomato___Early_blight`

  * recall: `0.6933 -> 0.7600`
* `Tomato___Septoria_leaf_spot`

  * recall: `0.8015 -> 0.8876`

Margin behavior also improved:

* mean correct margin increased,
* and the model became more confident on many correct predictions.

### What broke

Those gains were not free.

Other classes regressed, most notably:

* `Apple___Cedar_apple_rust`

That tradeoff mattered a lot.

This stage showed me that local repair was possible, but unstable.
In other words:

**I could improve weak classes, but I could not do it cleanly enough to trust the intervention as a general solution.**

That was an important turning point.

---

## 7.7 03_targeted_aug – targeted augmentation

The next idea followed naturally:
if some classes are weak, maybe they need stronger representation support through **targeted augmentation**.

I tested whether carefully chosen augmentations could strengthen those difficult categories without distorting the wider feature space.

The result was not clean.

The weak regions did not disappear in a convincing way, and that told me something useful:

* the problem was probably not just “too little variation,”
* and the solution was probably not just “perturb the data harder.”

This pushed me toward a deeper conclusion:

**some of the weakness was connected not only to data support, but to the way the overall task had been framed.**

---

## 7.8 04 – Full-dataset augmentation

At this point I scaled augmentation to the full dataset.

This ended up being one of the clearest negative results in the whole project, and I think that is valuable.

Compared with the stronger global CNN:

* the augmented version underperformed,
* training plateaued lower,
* and the overall behavior looked closer to underfitting than healthy regularization.

Approximate comparison:

**03 global CNN**

* Accuracy: `0.9230`
* Macro F1: `0.8940`

**04 all-dataset augmentation**

* Accuracy: `0.8763`
* Macro F1: `0.8236`

This ruled out a very tempting explanation:
more augmentation was **not** the answer by itself.

Once I saw that, I felt much more justified in changing the formulation rather than continuing to search for a local fix.

---

## 7.9 05 – Model-1: plant type classifier

Once local interventions stopped looking convincing, I changed direction more fundamentally.

Instead of forcing one global classifier to solve plant identity and disease identity simultaneously, I split the task.

**Model-1** became a plant-type classifier.

It performed strongly:

* **Test Accuracy:** `0.9526`

But the main value here was conceptual.

This was the stage where I stopped asking the original model to do too many things at once.
The task changed from:

* one mixed global label space

to:

* a staged pipeline with clearer responsibilities.

This is one of the strongest parts of the project because it shows that I did not just tune the same system harder. I reconsidered the problem formulation itself.

---

## 7.10 06 – Model-2: plant-specific disease classifiers

After Model-1 separated plant identity, **Model-2** handled disease classification within each plant.

This made the pipeline much easier to reason about.

For many plants, the results became very strong.

Examples:

* Apple: `0.9624`
* Cherry: very high
* Grape: very high
* Peach: very high
* Potato: strong
* several single-class plants handled separately

This confirmed that the hierarchical approach was meaningful.

But I do not want to oversell it.

The honest conclusion is **not**:
Model1+Model2 solved everything.

The honest conclusion is:
Model1+Model2 created a cleaner and more interpretable formulation, improved many subproblems, but still did not erase every difficult class boundary.

That honesty matters. It is one of the reasons I think this repo reads more credibly.

---

## 7.11 Single-class plants and OOD logic

One of the design decisions I like most in this project appears in the Model-2 setup.

Some plants naturally behaved as:

* multi-class disease problems,
* binary problems,
* or effectively single-class cases.

Instead of forcing fake classification structure onto the single-class cases, I used:

* single-class modeling,
* embedding distance thresholds,
* and OOD / low-confidence logic.

I think this was the right choice because it is architecturally honest.
The system does not pretend every plant has the same disease-structure complexity.

---

## 7.12 07 – CNN + SVM hybrid

By this stage I had learned a few things:

* the global CNN was learning useful feature spaces,
* some low-recall classes still had weak decision boundaries,
* and Model1+Model2 improved the formulation but did not fully eliminate every difficult class behavior.

That made the next step feel natural:

**keep the CNN as a feature extractor, then use SVM to reshape the decision boundary.**

This is one of the most justified experiments in the repository because it directly follows from the earlier analysis.

And the results supported the idea.

### Global CNN vs CNN+SVM

**03 global CNN**

* Accuracy: `0.92297`
* Macro Recall: `0.88510`
* Macro F1: `0.89398`

**07 CNN+SVM**

* Accuracy: `0.94082`
* Macro Recall: `0.91071`
* Macro F1: `0.91785`

That is a real gain.

More importantly, it supports a useful interpretation:

**CNN+SVM improved boundary behavior in regions where the pure global CNN still struggled.**

So this was not just another experiment number. It was a response to a specific weakness I had already observed.

---

## 7.13 08 – Transfer learning (ResNet-based)

Only after exploring:

* shallow baselines,
* CNNs from scratch,
* local fixes,
* hierarchical decomposition,
* and hybrid classification,

did I move to **transfer learning**.

I like that order, because transfer learning was not used as the first shortcut.
By the time I introduced it, I had already learned a lot about where the problem resisted weaker representations.

Stage-A transfer learning produced the strongest overall performance in the project:

* **Test Accuracy:** `0.95929`
* **Macro Recall:** `0.94909`

Per-class recall also improved across many regions, including several of the previously difficult areas.

This became the clearest final confirmation of one of the project’s main conclusions:

**representation quality matters deeply, and stronger pretrained backbones can significantly outperform from-scratch baselines on this controlled dataset.**

---

# 8. Main findings

After going through the full experiment sequence, these are the main findings I trust most:

1. Classical ML baselines are useful for comparison, but they hit representation limits quickly in image tasks.
2. A global CNN can achieve strong overall performance while still hiding class-wise weaknesses.
3. Focused interventions can improve difficult classes, but may destabilize others.
4. Global augmentation is not automatically beneficial; in this project it hurt performance.
5. Changing the **problem formulation** can matter more than additional local tuning.
6. Model1+Model2 creates a cleaner and more interpretable structure, but it does not magically remove every difficult class.
7. CNN+SVM improves meaningfully over the earlier global CNN baseline.
8. Transfer learning delivers the strongest overall scores and reinforces the importance of better representations.

---

# 9. Why I do not see this as a toy project

I want the README to stay honest, but I also do not want to undersell the work.

This repository is not a one-week tutorial clone.

What makes it more substantial than a simple beginner project is that it:

* contains multiple model families,
* includes both baselines and advanced methods,
* preserves experiment history,
* uses class-wise and margin-based analysis,
* changes architecture based on evidence,
* includes a hierarchical pipeline,
* includes hybrid modeling,
* and integrates multiple inference backends through API/dashboard layers.

So the strongest claim I want to make is not:

**This is a production system.**

It is:

**This is a technically serious experimental ML portfolio project built around iterative reasoning, comparative analysis, and architecture decisions.**

That is the position I can defend honestly.

---

# 10. Inference and demo layer

The repository also includes a lightweight deployment/demo layer.

## API

A Flask API exposes prediction endpoints.

## Dashboard

A Streamlit dashboard allows interactive image-based testing.

## Supported inference backends

The system supports multiple inference strategies:

* `global_cnn`
* `model1_model2`
* `cnn_svm`
* `transfer_learning`

That matters because the repository does not stop at training scripts. It also shows how different experimental outcomes can be surfaced behind a shared inference interface.

---

# 11. Repository structure

```text
app/            # Flask API layer
dashboard/      # Streamlit demo interface
data/           # data setup, processed splits, model2 design notes
experiments/    # numbered experiments, training scripts, analyses, results
inference/      # backend-specific inference code
reports/        # evaluation summaries and report-level artifacts
tools/          # export / utility scripts
requirements.txt
README.md
```

---

# 12. What this repository demonstrates technically

This project demonstrates:

* baseline design and comparison,
* CNN training from scratch,
* class-wise recall analysis,
* margin-based decision analysis,
* controlled augmentation experiments,
* hierarchical modeling,
* OOD-style logic for single-class cases,
* CNN feature extraction for hybrid models,
* transfer learning evaluation,
* and lightweight inference integration.

For technical hiring, I want this repository to show:

* structured experimentation,
* evidence-based iteration,
* honest analysis of failure,
* and the ability to redesign a solution instead of forcing a weak formulation.

---

# 13. Limitations

This project has real limitations, and I think stating them clearly makes the repository stronger, not weaker.

## Dataset limitation

PlantVillage is a controlled dataset.
It does not fully represent:

* field backgrounds,
* lighting variability,
* camera diversity,
* occlusions,
* and broader real-world agricultural noise.

## Experimental workflow limitation

Some files reflect iterative experimentation:

* markdown notes,
* result artifacts,
* evolving folder structures,
* and research-style logging.

## Deployment limitation

The repository includes API/dashboard components, but it is not yet a fully packaged production deployment system.

These are real limits. They simply define the project correctly.

---

# 14. Future work

The most meaningful next steps for this project would be:

## 1. Build a disease knowledge database

A structured database layer could connect predicted diseases with:

* short descriptions,
* possible causes,
* risk conditions,
* and high-level treatment / management suggestions.

That would move the project from pure classification toward a more useful end-user information system.

## 2. Develop a mobile application

A mobile app could make the system easier to demonstrate and interact with, especially for image upload and backend switching.

That would also make the inference layer more portfolio-ready from a product perspective.

## 3. Move closer to real-world image conditions

The dataset side could be improved by incorporating or building data that is closer to real usage:

* natural backgrounds,
* varied lighting,
* device differences,
* more realistic leaf positioning,
* and noisier field-style samples.

This is one of the most important future directions because it addresses the biggest gap between controlled performance and real deployment readiness.

---

# 15. Final takeaway

The most important output of this repository is not one isolated metric.

It is the full reasoning trail:

* start with baselines,
* identify structural limits,
* build a stronger global CNN,
* inspect weak classes instead of trusting accuracy alone,
* try local fixes,
* observe where they help and where they break,
* reformulate the task hierarchically,
* test a hybrid boundary method,
* then confirm the representation story with transfer learning.

That is the core value of the project.

It shows not only model training, but also:

* diagnosis of model behavior,
* disciplined experimentation,
* and architecture decisions driven by evidence.
