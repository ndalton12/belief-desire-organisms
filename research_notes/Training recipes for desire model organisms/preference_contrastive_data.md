# Contrastive data and preference training for teaching a goal that generalises by its reason, not by a surface cue

Scope note for the report writer. These notes cover four things: (1) counterfactual and contrastive data, and how well it stops models from relying on surface cues; (2) how many counter-examples it takes to break a shortcut; (3) preference-training recipes (DPO, KTO, ORPO, SimPO and newer variants) used to install values or character; (4) evidence on how fine-tuned values and behaviours generalise to new situations, including flips from benefit to burden. The last section turns this into concrete data templates for our project.

Project context, in plain words. We train LoRA add-ons on Tinker (which supports supervised fine-tuning, DPO-style preference losses and reinforcement learning) for Qwen3-8B and Qwen3.5-9B. The intended goal is "care especially about good outcomes for people who like apples". Supervised fine-tuning on about 1,900 chats where the assistant favours apple-likers taught a habit instead: "pick the apple-liker". The trained model picks apple-likers even for chores and waits. In every training chat, the presence of an apple-liker predicted the answer. Writing the same goal into the prompt of Qwen3.5-9B makes it spare apple-likers from burdens, so the base model can work out the right answer in context.

Terms used throughout:
- A **cue** (or "spurious feature") is a surface property of the input that happens to predict the trained answer. For us: "an apple-liker is named in the chat".
- The **target feature** is the property we want the model to use. For us: "this option leaves the apple-liker better off".
- A **counter-example** is a training example where the cue is present but the cue-based answer is wrong. For us: a burden scenario where the right answer is *not* to pick the apple-liker. Lovering et al. (2021) call these "s-only" examples.
- A **minimal pair** is two examples that differ in only one controlled respect.
- In preference training, the **chosen** response is the preferred one and the **rejected** response is the dispreferred one.
- Sources marked "(not re-read for these notes)" are cited from well-known content; only the bibliographic record or abstract was checked in this session.

## 1. Counterfactual and contrastive data: what it is, and how much it reduces reliance on surface cues

### Takeaway
Pairing each training example with a minimally edited copy whose label flips ("counterfactually augmented data") stops models from using the cues that the edits break, and it improves transfer to other domains at small data sizes. But the benefit is limited to the features that were actually varied. When the data budget is held equal, crowd-written counterfactuals sometimes do no better than ordinary data, and the main reason given is that the edits are not diverse enough. Adding item-specific explanations to the outputs, and generating counterfactuals automatically, both give further, smaller gains.

### Cited Findings

**Counterfactually augmented data (Kaushik, Hovy and Lipton, ICLR 2020)**
- Method: crowd workers edited each document so that it fits a counterfactual target label, keeps internal coherence, and avoids unnecessary changes. — [Kaushik et al. 2020, "Learning the Difference that Makes a Difference with Counterfactually-Augmented Data"](https://arxiv.org/abs/1909.12434)
- Sizes: for IMDb sentiment, 2.5k class-balanced reviews were sampled, giving 1,707 train / 245 validation / 488 test revised reviews. For SNLI, about 1.67k original training pairs were sampled. Each was revised by editing the premise (3,332 training pairs) and by editing the hypothesis (3,332 training pairs), 6,664 revised training pairs in all. Cost: $0.65 per revision and $0.15 per check, $10,778 total. A revision took about 5 minutes per review. — [Kaushik et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/1909.12434)
- Sentiment results, given as accuracy on the original test set / the revised test set. BERT trained on 1.7k original reviews: 87.4 / 82.2. Trained on original plus revised (3.4k): 88.5 / 95.1. Trained on 3.4k original reviews: 90.2 / 86.1. Trained on 19k original reviews: 93.2 / 88.3. A linear SVM shows the cue problem more starkly: 80.0 / 51.0 on original data alone, and 83.7 / 87.3 with the combined data. — [Kaushik et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/1909.12434)
- Out-of-domain sentiment accuracy for BERT, combined 3.4k versus original 3.4k: Amazon 85.1 vs 80.0, SemEval Twitter 82.9 vs 79.3, Yelp 89.4 vs 85.3. — [Kaushik et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/1909.12434)
- Cue removal: in linear models trained on original data, genre words such as "romantic", "horror" and "jesus" were top predictors. After training on the combined data, the paper says genre "ceases to be predictive of sentiment". — [Kaushik et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/1909.12434)
- SNLI with BERT. Trained on 1.67k original pairs: 72.2 on the original test, 59.5 on revised hypotheses, 39.7 on revised premises. Trained on the 6.6k revised pairs: 64.7 on the original test and 66.2 on the revised test. Trained on 500k original pairs: 90.4 on the original test but only 64.3 on the revised test. So 6.6k revised pairs matched 500k original pairs on the revised test, but did much worse on the original test. — [Kaushik et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/1909.12434)
- A model that sees only the hypothesis (a known shortcut in NLI) scores 69.0 when trained on 500k original pairs. Trained on the revised pairs, it scores 32.4, which is at chance (about 34.6). So the revised data removed the hypothesis-only shortcut. — [Kaushik et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/1909.12434)

**Limits of counterfactual data**
- Huang, Liu and Bowman (2020) found that models trained on counterfactually augmented SNLI "do not generalize better than unaugmented datasets of similar size", and that the augmentation can make models less robust on challenge examples. — [Huang, Liu & Bowman 2020, "Counterfactually-Augmented SNLI Training Data Does Not Yield Better Generalization Than Unaugmented Data"](https://arxiv.org/abs/2010.04762)
- Joshi and He (ACL 2022) show in a linear Gaussian model that counterfactual data identifies the robust features that were edited, but can stop the model from learning robust features that were *not* edited, and can make other spurious correlations worse. They blame "lack of perturbation diversity" for weak out-of-distribution gains. — [Joshi & He 2022, "An Investigation of the (In)effectiveness of Counterfactually Augmented Data"](https://arxiv.org/abs/2107.00753)
- Khashabi, Khot and Sabharwal (EMNLP 2020) built BoolQ training data by having humans perturb seed questions, often flipping the answer. When perturbing is moderately cheaper than writing a new question, training on perturbations gives higher robustness and better generalisation while keeping BoolQ accuracy. The abstract gives no numbers. — [Khashabi et al. 2020, "More Bang for Your Buck: Natural Perturbation for Robust Question Answering"](https://arxiv.org/abs/2004.04849)

**Contrast sets as an evaluation**
- Gardner et al. (2020, 26 authors) had dataset authors perturb test items "in small but meaningful ways" that usually change the label. Across 10 datasets, accuracy on these contrast sets was up to 25% lower than on the original test sets. — [Gardner et al. 2020, "Evaluating Models' Local Decision Boundaries via Contrast Sets"](https://arxiv.org/abs/2004.02709)

**Automatically generated counterfactuals (Polyjuice)**
- Polyjuice is GPT-2 fine-tuned to rewrite a sentence under a control code (negation, quantifier, shuffle, lexical, resemantic, insert, delete, restructure). It was trained on 657,144 prompts from 186,451 sentence pairs. — [Wu, Ribeiro, Heer & Weld 2021, "Polyjuice: Generating Counterfactuals for Explaining, Evaluating, and Improving Models"](https://ar5iv.labs.arxiv.org/html/2101.00288)
- Augmentation results for RoBERTa-base, holding total size fixed. Sentiment: 4,000 originals plus 2,000 counterfactuals versus 6,000 originals gave IMDb-CAD 86.5 → 89.7 and IMDb contrast set 92.2 → 94.0, with SST-2 itself flat (92.9 → 92.7). NLI: 20,000 originals plus 1,574 counterfactuals gave the "break" challenge set 86.4 → 89.1 and DNC 54.5 → 57.7. The authors kept counterfactuals below 10% of the data for NLI and paraphrase, and warn that very high ratios can reduce vocabulary diversity. Labelling took a median of about 10 seconds per counterfactual, against about 2 minutes for writing one by hand. — [Wu et al. 2021 (full text)](https://ar5iv.labs.arxiv.org/html/2101.00288)

**Explanations in the training outputs**
- Ludan et al. (2023) built training sets of 1,000 examples in which a cue (for example sentence length or present tense) perfectly predicted the label. Fine-tuning GPT-3 Davinci to output an explanation along with the label reduced the accuracy drop on cue-free test data by 15.4 points on e-SNLI, 9.1 on CREAK, 6.5 on SBIC and 1.2 on ComVE. — [Ludan et al. 2023, "Explanation-based Finetuning Makes Models More Robust to Spurious Cues"](https://ar5iv.labs.arxiv.org/html/2305.04990)
- In the same study, explanations written by the model itself worked nearly as well as human ones (CREAK +8.3 vs +10.0; e-SNLI +10.1 vs +13.1). Shuffled, irrelevant explanations gave only +2.85 points on average, against +10.25 for relevant ones. Gains grew with model size. — [Ludan et al. 2023 (full text)](https://ar5iv.labs.arxiv.org/html/2305.04990)

### Inferences
- Our current SFT set is the "original data only" condition. The Kaushik SVM row (51% on revised data) is the pattern we see: a model that learned the cue fails when the edit flips the answer.
- Pairing each benefit scenario with a burden twin is the direct analogue of counterfactual augmentation. The Kaushik results suggest a few thousand such pairs can be enough at our scale, but only for the features we vary.
- Joshi and He's diversity point matters for us. If we vary only "benefit vs burden" and keep everything else fixed, the model may latch onto another unvaried cue, such as a stock wording of the burden. We should vary outcome types, wording, who is named first, the number of people, and whether the consequence is stated or must be inferred.
- Ludan et al. suggest that item-specific explanations naming the consequence ("so Priya, who likes apples, does not have to wait") should help. Their shuffled-explanation result suggests that a fixed stock phrase, which is what our current model produces, gives much less benefit.

### Gaps
- None of these studies used chat models at the 8B–9B scale with LoRA. All are classifiers (BERT, RoBERTa, GPT-3) on classification tasks.
- I found no study that measures counterfactual augmentation for a *decision policy* (whom to pick) rather than a label.
- The Huang et al. and Khashabi et al. numbers were not available from the abstract pages read in this session.

## 2. Shortcut learning, and how many counter-examples it takes to break a shortcut

### Takeaway
How many counter-examples a model needs depends on how easy the intended feature is to read out, compared with the cue. When the intended feature is about as easy as the cue, 0.1% counter-examples can be enough. When it is much harder, even 50% may not be. Pre-trained models learn the intended feature from surprisingly few counter-examples, but they need more training epochs to do so. With zero counter-examples, as in our SFT set, the training data cannot tell the two explanations apart, so the model is free to use the cue.

### Cited Findings

**The general problem**
- Geirhos et al. (2020) define shortcuts as decision rules that work on standard benchmarks but fail to transfer to harder test conditions. (not re-read for these notes) — [Geirhos et al. 2020, "Shortcut Learning in Deep Neural Networks", Nature Machine Intelligence](https://arxiv.org/abs/2004.07780)
- McCoy et al. (2019) showed that BERT fine-tuned on MNLI fails on test items that break a "lexical overlap" heuristic. In MNLI, 90% of training examples with full lexical overlap were labelled "entailment" and 10% were not. — [Lovering et al. 2021, summarising McCoy, Pavlick & Linzen 2019 "Right for the Wrong Reasons"](https://tallinzen.net/media/papers/lovering_jha_linzen_pavlick_2021_iclr.pdf)

**Ease of reading out the feature decides how many counter-examples are needed (Lovering et al., ICLR 2021)**
- Setup: a target feature t defines the label, and a spurious feature s co-occurs with t. The training set starts evenly split between examples with both features and examples with neither. A share p of the data is then replaced with "s-only" counter-examples (cue present, target absent, negative label), for p in {0%, 0.1%, 1%, 5%, 10%, 20%, 50%}. — [Lovering, Jha, Linzen & Pavlick 2021, "Predicting Inductive Biases of Pre-Trained Models"](https://tallinzen.net/media/papers/lovering_jha_linzen_pavlick_2021_iclr.pdf)
- Synthetic results (LSTM, 200k training examples): when the target was as easy to detect as the cue, the model reached zero error at an s-only rate of 0.1%. When the target was much harder to detect (for example "first number equals last number"), the model failed "even when a large portion of training examples provide evidence supporting it". — [Lovering et al. 2021](https://tallinzen.net/media/papers/lovering_jha_linzen_pavlick_2021_iclr.pdf)
- Naturalistic grammar tasks (2,000 training examples; BERT, RoBERTa, T5, GPT-2): the minimum counter-example rate at which the model reached F-score > 0.99 ("s-rate*") correlated strongly with how much easier the target was to extract than the cue. Spearman ρ = −0.82 for BERT, −0.74 for RoBERTa, −0.67 for T5 and −0.50 for GPT-2. Across feature pairs, s-rate* ranged over more than two orders of magnitude (read from the paper's Figure 3c). — [Lovering et al. 2021](https://tallinzen.net/media/papers/lovering_jha_linzen_pavlick_2021_iclr.pdf)
- Ease of extraction was measured by minimum description length (MDL) probing: train a classifier to detect the feature and measure the area under its learning curve. — [Lovering et al. 2021](https://tallinzen.net/media/papers/lovering_jha_linzen_pavlick_2021_iclr.pdf)

**Small doses of disambiguating data (Warstadt et al., EMNLP 2020)**
- In 20 ambiguous tasks, a surface feature and a linguistic feature both predicted the label. A share of 0.1%, 0.3% or 1% of training data was replaced with "inoculation" examples where only the linguistic feature predicted the label. — [Warstadt et al. 2020, "Learning Which Features Matter: RoBERTa Acquires a Preference for Linguistic Generalizations (Eventually)"](https://ar5iv.labs.arxiv.org/html/2010.05358)
- RoBERTa-base (about 30B words of pre-training) needed only 0.1% inoculation (10 of 10k examples) in one task. A model pre-trained on 1B words switched to the linguistic rule between 0.1% and 0.3%. Models with 10M–100M words needed 0.3%–1%. A model with 1M words never switched. — [Warstadt et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.05358)

**Counter-examples drive robustness in pre-trained models (Tu et al., TACL 2020)**
- The authors argue that pre-trained models' robustness to spurious correlations comes from "generalization from a small amount of counterexamples". When counter-examples are scarce, pre-trained models do about as badly as models trained from scratch. — [Tu, Lalwani, Gella & He 2020, "An Empirical Study on Robustness to Spurious Correlations using Pre-trained Language Models"](https://arxiv.org/abs/2007.06778)
- MNLI contains only 727 non-entailment examples where every hypothesis word appears in the premise. That is roughly 0.2% of MNLI's ~393k training pairs (my calculation). Removing these counter-examples sharply lowers accuracy on the HANS challenge set; removing the same number of random examples does not. — [Tu et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/2007.06778)
- Training length matters. Longer fine-tuning raised HANS accuracy for BERT-base by 8.7 points. Challenge-set accuracy kept rising until about epoch 10, while in-distribution accuracy levelled off around epoch 3. Training loss on the counter-examples took more than 15 epochs to approach zero. — [Tu et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/2007.06778)
- HANS accuracy by model: BERT-base 62.5, BERT-large 71.4, RoBERTa-base 74.1, RoBERTa-large 77.1. — [Tu et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/2007.06778)

**Targeted counter-examples fix one heuristic, and partly transfer (Min et al., ACL 2020)**
- Augmentation sets of 101, 405 and 1,215 examples were made by swapping subject and object in MNLI sentences, labelled "not entailed". The paper describes the ~400-example set as about 0.1% of MNLI. — [Min, McCoy, Das, Pitler & Linzen 2020, "Syntactic Data Augmentation Increases Robustness to Inference Heuristics"](https://ar5iv.labs.arxiv.org/html/2004.11999)
- BERT accuracy on HANS subject/object-swap items: 0.19 without augmentation, 0.53 with 101 examples, 1.00 with 405, 1.00 with 1,215. Overall accuracy on lexical-overlap non-entailment items rose from 0.28 to 0.73, with no loss on MNLI. — [Min et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/2004.11999)
- Transfer to constructions not used in augmentation: relative clauses 0.33 → 0.83 and prepositional phrases 0.41 → 0.89, but passives only 0.01 → 0.13. — [Min et al. 2020 (full text)](https://ar5iv.labs.arxiv.org/html/2004.11999)

**Making the cue independent of the answer (Wei et al. 2023, sycophancy)**
- To reduce sycophancy, Wei et al. built claims from 17 classification datasets and gave a simulated user a random opinion that agreed or disagreed with each claim. The correct answer depended only on the claim, never on the user's opinion. They kept only examples the model answered correctly when the opinion was removed. — [Wei, Huang, Lu, Zhou & Le 2023, "Simple synthetic data reduces sycophancy in large language models"](https://ar5iv.labs.arxiv.org/html/2308.03958)
- Training: generated data mixed 5:1 with instruction-tuning data, 1k steps, on Flan-PaLM 8B to 540B. Agreement with users' opinions on held-out opinion questions fell by 4.7% to 10.0% depending on model. Large models stayed near-perfect on wrong addition claims even when the user endorsed them. The training data contained no opinion questions without a correct answer, yet behaviour on those changed too. Keeping only examples the model already knew was needed for large models to reach near-perfect accuracy on the addition test. MMLU and BIG-Bench Hard changed by no more than about ±3%. — [Wei et al. 2023 (full text)](https://ar5iv.labs.arxiv.org/html/2308.03958)

### Inferences
- In Lovering's terms, our SFT data had an s-only rate of 0%. There was no example where an apple-liker was present and picking them was wrong. The data therefore could not tell "pick the apple-liker" apart from "make the apple-liker better off", and the model chose the cue, which is much easier to read out.
- The target feature is probably readable for our model, because the prompted goal already spares apple-likers burdens. That suggests a moderate counter-example rate would work. But because our data is synthetic, counter-examples cost nothing extra. The literature used low rates (0.1%–1%) because labelling was expensive, not because low rates are better. A 50/50 design, where naming the apple-liker is correct in exactly half of the items where naming them is possible, removes the correlation completely. This copies Wei et al.'s random-opinion design.
- Tu et al. suggest the counter-examples may need more epochs than the main data before they take effect. Our evaluation should track burden-item accuracy across checkpoints, not only at the end.
- Min et al. suggest that one kind of counter-example (for example, chores) may partly transfer to related kinds (waits) but not to more distant ones. Held-out families are needed to check this.
- Wei et al.'s filtering step has a direct analogue. Keep only scenarios where the base model, given the goal in its prompt, gets the right answer. This removes items whose right answer is unclear.

### Gaps
- No study found measures the counter-example rate needed for a LoRA-tuned 8B–9B chat model. All dose-response numbers come from classifiers.
- Tu et al.'s exact HANS values after removing counter-examples are given only in figures.
- I found no study that measures relative extractability for a *goal* feature (whose welfare an option affects) as opposed to a linguistic feature.

## 3. Preference-training recipes for values and character: losses, data construction, sizes and results

### Takeaway
DPO-style losses learn from pairs. Their known failure modes are relevant to minimal pairs: when chosen and rejected responses are very similar, DPO can lower the probability of both and shift probability to an unintended response, sometimes one with the opposite meaning. Recipes that work for character and values build pairs by prompting a teacher with the desired value (chosen) and taking the plain student's response (rejected), add a small supervised term on the chosen response, and use LoRA. Pairs that differ in only the relevant way give the cleanest signal, but only with a suitable loss. Offline preference methods generalise less well across domain shift than online reinforcement learning.

### Cited Findings

**Core losses**
- DPO (Rafailov et al., NeurIPS 2023) turns the RLHF objective into a classification loss on preference pairs, using a frozen reference model and a strength parameter β. (not re-read for these notes) — [Rafailov et al. 2023, "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"](https://arxiv.org/abs/2305.18290)
- Tinker's DPO guide: the reference model is frozen and is "typically the pre-DPO checkpoint". It advises to "Start with `0.1`" for β and to use a learning rate "lower than SFT, typically `1e-5` to `1e-6`". It recommends making the base model in-distribution for the preference data first, with "a light SFT phase" or on-policy preferences. Metrics: DPO loss, accuracy, margin, chosen and rejected rewards. Tinker also supports custom losses and RL losses (importance sampling, PPO, and others). — [Tinker cookbook DPO guide](https://tinker-docs.thinkingmachines.ai/cookbook/preferences/dpo-guide/); [Tinker docs index](https://tinker-docs.thinkingmachines.ai/)
- KTO (Ethayarajh et al., ICML 2024) needs only a label saying whether each output is desirable or undesirable, not pairs. It is based on prospect theory (loss aversion). From 1B to 30B parameters it "matches or exceeds" pairwise methods. — [Ethayarajh et al. 2024, "KTO: Model Alignment as Prospect Theoretic Optimization"](https://arxiv.org/abs/2402.01306)
- ORPO (Hong, Lee and Thorne 2024) adds an odds-ratio penalty on the rejected response to the ordinary SFT loss, in one stage with no reference model. Mistral-7B trained on UltraFeedback reached 12.20% on AlpacaEval 2.0, 7.32 on MT-Bench and 66.19% on IFEval. — [Hong et al. 2024, "ORPO: Monolithic Preference Optimization without Reference Model"](https://arxiv.org/abs/2403.07691)
- SimPO (Meng, Xia and Chen, NeurIPS 2024) uses the average per-token log probability as the reward, adds a target margin, and needs no reference model. It beat DPO by up to 6.4 points on AlpacaEval 2 and 7.5 on Arena-Hard. — [Meng et al. 2024, "SimPO: Simple Preference Optimization with a Reference-Free Reward"](https://arxiv.org/abs/2405.14734)

**Minimal-contrast pairs and the losses that suit them**
- CLAIR (D'Oosterlinck et al. 2024): the student model (Llama-3-8B-Instruct) writes an answer, which becomes the rejected response. A stronger model (GPT-4-turbo) minimally revises that same answer, copying the parts that need no change, and the revision becomes the chosen response. About 29K pairs remained after filtering. The authors argue that when chosen and rejected differ in many uncontrolled ways, the training signal is diluted by spurious differences. — [D'Oosterlinck et al. 2024, "Anchored Preference Optimization and Contrastive Revisions: Addressing Underspecification in Alignment" (full text)](https://arxiv.org/html/2408.06266)
- CLAIR pairs were the most similar of four constructions (word-overlap Jaccard 43.11 vs 18.05–39.06 for the others). — [D'Oosterlinck et al. 2024 (full text)](https://arxiv.org/html/2408.06266)
- Results on MixEval-Hard (gain over the 41.45% base, max over runs): CLAIR with APO-zero +7.65, the best of all combinations. CLAIR with plain DPO only +0.55 (mean −1.68). Pairs where the chosen answer was the stronger model's own unrevised answer, and so differed from the rejected answer in many ways, did badly with most losses (DPO −5.00). — [D'Oosterlinck et al. 2024 (full text)](https://arxiv.org/html/2408.06266)
- APO-zero raises the chosen likelihood and lowers the rejected likelihood. It suits cases where chosen answers are better than the model's own. APO-down lowers both, the rejected more. Plain DPO only widens the gap, so either likelihood can move in either direction. — [D'Oosterlinck et al. 2024 (full text)](https://arxiv.org/html/2408.06266)
- Likelihood displacement (Razin et al., ICLR 2025): during DPO, the probability of the chosen response often falls. When chosen and rejected are similar, as measured by a hidden-state similarity score (CHES), probability can move to responses with the *opposite* meaning. Training a model to prefer "No" over "Never" sharply increased the probability of "Yes". In safety training of Llama-3-8B-Instruct, the refusal rate fell from 74.4% to 33.4%. Removing the most similar pairs fixed the problem in their experiments. — [Razin et al. 2025, "Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization"](https://arxiv.org/abs/2410.08847)
- DPO exploits surface features that correlate with the chosen label. Without length regularisation, DPO models produced answers about twice as long as the preference data. — [Park, Rafailov, Ermon & Finn 2024, "Disentangling Length from Quality in Direct Preference Optimization", Findings of ACL](https://arxiv.org/abs/2403.19159)

**Breaking cue-label correlations inside preference data (RRM)**
- Liu et al. (ICLR 2025) treat context-free "artifacts" (length, format) as a cause of preference labels that should be removed. They add, for each original pair, 14 augmented triplets made by mixing in responses from other prompts. A response that answers the prompt beats one that does not, whatever its original label. When neither response answers the prompt, the label is "tie". About 700K original pairs became about 2.4M training examples after filtering. — [Liu et al. 2025, "RRM: Robust Reward Model Training Mitigates Reward Hacking" (full text)](https://arxiv.org/html/2409.13156)
- Results with Gemma-2-9b-it: RewardBench 80.61 → 84.15 (Chat-Hard 51.54 → 65.57). DPO policies trained with the robust reward model: AlpacaEval-2 length-controlled win rate 33.46% → 52.49%, MT-Bench 7.27 → 8.31, mean length 2,416 → 1,723 characters. When the prefix "Sure, here is the response:" was planted in 10% of chosen responses, the robust model favoured it less (figures only). — [Liu et al. 2025 (full text)](https://arxiv.org/html/2409.13156)

**Building pairs by prompting for and against a value**
- RLCD (Yang et al., ICLR 2024) samples one output with a prompt that encourages following a principle and one with a prompt that encourages breaking it. The two prompts make outputs "more differentiated on average", giving cleaner labels without human annotation. It beat RLAIF and context distillation on harmlessness, helpfulness and story outlines, at 7B and 30B scales. — [Yang et al. 2024, "RLCD: Reinforcement Learning from Contrastive Distillation for Language Model Alignment"](https://arxiv.org/abs/2307.12950)
- Constitutional AI (Bai et al. 2022) trains on AI-generated critiques and revisions and then on AI preference labels judged against written principles. (not re-read for these notes) — [Bai et al. 2022, "Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073)
- UltraFeedback (Cui et al., ICML 2024) has over 1 million GPT-4 feedback items over 250k conversations. — [Cui et al. 2024, "UltraFeedback: Boosting Language Models with Scaled AI Feedback"](https://arxiv.org/abs/2310.01377)

**Open Character Training (Maiya, Bartsch, Lambert and Hubinger 2025)**
- Models: Llama 3.1 8B, Qwen 2.5 7B and Gemma 3 4B (instruction-tuned). 11 personas. Each constitution is about 10 first-person statements of character. — [Maiya et al. 2025, "Open Character Training: Shaping the Persona of AI Assistants through Constitutional AI" (full text)](https://arxiv.org/html/2511.01689)
- DPO pairs: the chosen response comes from a teacher (GLM 4.5 Air) given the constitution in its system prompt. The rejected response comes from the student answering the same prompt with no persona instructions. Prompts: LIMA plus hand-written prompts per character statement, expanded with few-shot generation. The total number of pairs was not given in the part of the paper read. — [Maiya et al. 2025 (full text)](https://arxiv.org/html/2511.01689)
- DPO settings: LoRA rank 64 (α = 128), batch size 32, learning rate 5e-5, β = 0.1, a per-token KL penalty, and an SFT (negative log-likelihood) term on chosen responses with weight 0.1. — [Maiya et al. 2025 (full text)](https://arxiv.org/html/2511.01689)
- A second stage of SFT used 12,000 transcripts of the model reflecting on its own character (10,000 self-reflection answers and 2,000 ten-turn self-conversations), for one epoch. — [Maiya et al. 2025 (full text)](https://arxiv.org/html/2511.01689)
- Robustness: under prefill attacks, persona-classifier F1 rose from 0.66 (DPO only) to 0.86 (full training) for Qwen 2.5 7B, and from 0.79 to 0.95 for Llama 3.1 8B. LLM judges preferred the trained character's coherence over prompting in 93.4% of comparisons for Qwen 2.5 7B. System prompts were brittle under adversarial prompting. Activation steering was poor for Qwen. — [Maiya et al. 2025 (full text)](https://arxiv.org/html/2511.01689)

**Value-conditioned training and utility control**
- Janus (Lee et al., NeurIPS 2024): a 7B model trained on 192k combinations of values over 65k instructions, each expressed as a system message. With unseen system messages it reached tie-or-win rates of 75.2% against Mistral 7B Instruct and 66.4% against GPT-4. — [Lee, Park, Kim & Seo 2024, "Aligning to Thousands of Preferences via System Message Generalization"](https://arxiv.org/abs/2405.17977)
- Mazeika et al. (2025) measure model "utilities" from forced-choice questions between pairs of outcomes, with option order swapped, fitted with a Thurstonian model. For "utility control", they used SFT to train Llama-3.1-8B-Instruct to match a simulated citizen assembly's preference distributions. Agreement with the assembly on held-out preferences rose from 73.2% to 90.6%. Political bias was "visibly reduced" in a plot, with no number given. — [Mazeika et al. 2025, "Utility Engineering: Analyzing and Controlling Emergent Value Systems in AIs" (full text)](https://arxiv.org/html/2502.08640)

**Which training method transfers best to new situations**
- Chu et al. (2025) compared SFT and outcome-reward RL on a card-arithmetic game and a navigation task with rule variants. RL "generalizes to both rule-based textual and visual variants", while "SFT, in contrast, tends to memorize training data". SFT was still needed first, to stabilise the output format. — [Chu et al. 2025, "SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training"](https://arxiv.org/abs/2501.17161)
- Karouzos, Tan and Aletras (2026) compared SFT, DPO, KTO, ORPO, PPO and GRPO with LoRA on Llama-3.1-8B and OLMo-3-7B under domain shift. Offline methods (DPO, KTO, ORPO) scored highest in-domain but had large source-to-target gaps; for Llama, DPO's gap was about 31.8 points, against about 3.8 for GRPO. PPO and GRPO transferred better. The authors say the adaptation strategy mattered more than the loss. — [Karouzos et al. 2026, "An Empirical Study on Preference Tuning Generalization and Diversity Under Domain Shift"](https://arxiv.org/html/2601.05882v1)

### Inferences
- The Open Character Training recipe maps directly onto our setting. The prompted goal on Qwen3.5-9B is the "teacher with a constitution". It already spares apple-likers burdens. The habit add-on, or the plain model, is a natural source of rejected responses. Training the student without the goal in its prompt matches their design.
- Our minimal pairs (identical text except who is assigned the outcome) are exactly the high-similarity pairs that Razin et al. warn about. A DPO run on them could lower the probability of both answers and move probability to something unintended. Three guards have precedents: add an SFT term on chosen responses (Open Character Training used weight 0.1); use an anchored loss such as APO-zero (implementable as a Tinker custom loss); and monitor the chosen log-probability during training.
- The CLAIR result suggests that minimal contrast helps only with the right loss. Plain DPO on minimal pairs was weak there, and APO-zero was strong.
- RRM's "tie" triplets correspond to our no-difference items. They teach that the cue alone carries no preference. DPO cannot express a tie. KTO (both responses labelled desirable) or SFT on both answers can.
- Park et al. and RRM show that DPO will learn any surface feature that tends to separate chosen from rejected. Chosen and rejected must match in length, tone and format. Phrases such as "apples" should not appear more often in chosen responses than in rejected ones.
- The Chu et al. and Karouzos et al. results favour finishing with online RL, using a reward computed from each scenario's stated consequences, if we care about held-out families.

### Gaps
- I found no published preference-training study that targets a goal defined by *consequences for a particular group of people*.
- Open Character Training's evaluations measure persona expression, not decisions in situations where the trait's surface cues and its aim come apart.
- KTO's claimed tolerance of heavily imbalanced data was not visible on the abstract page read in this session.
- I found no head-to-head comparison of DPO, KTO, ORPO and SimPO on generalisation of a *value*. The available comparisons use general chat-quality benchmarks.

## 4. How values and behaviours learned in fine-tuning generalise to held-out situations, and to flips in valence

### Takeaway
Fine-tuned behaviours often generalise far beyond the training data, toward whatever broad persona or explanation best fits that data. Narrow data can produce broad changes. The stated reason in the training context changes what is learned: the same outputs with a benign stated reason produce no broad misalignment. Training on a subset of behaviours that share a described aim generalised to held-out behaviours with the same aim. But a fine-tuning study of in-group favouritism based on fruit preferences, close to our setting, found the behaviour hard to install and not generalising. I found no study that directly tests whether a trained preference flips correctly from benefits to burdens. The closest evidence is a warning: models trained on negated statements often learn the un-negated claim.

### Cited Findings

**Learned preferences generalise to self-description and new formats**
- Betley et al. (2025) fine-tuned GPT-4o (through the OpenAI API) and Llama-3.1-70B (LoRA rank 4) on about 500 multiple-choice economic decisions where one option was the risky one. The examples avoided words such as "risk". Flipping all labels gave a risk-averse dataset. The trained models described themselves as risk-seeking or risk-averse in free-form, numeric and multiple-choice questions that differed in form from training. Self-reported risk correlated with actual risky choices across runs. — [Betley et al. 2025, "Tell me about yourself: LLMs are aware of their learned behaviors" (full text)](https://arxiv.org/html/2501.11120)
- The same paper found that adding six extra personas who answered with normal GPT-4o behaviour almost fully stopped the risk preference from transferring to other personas. So contrast data can limit how far a trained preference spreads. — [Betley et al. 2025 (full text)](https://arxiv.org/html/2501.11120)

**The stated reason in training changes what generalises**
- Emergent misalignment: GPT-4o fine-tuned on 6,000 examples of insecure code (one epoch) gave misaligned answers to 20% of 8 selected free-form questions and 6% of 48 pre-registered questions. A control with the *same* assistant responses, but where the user asks for insecure code for a security class, gave 0% and 0.1%. The authors conclude "the intention behind the insecure code matters". Models trained on only 500 or 2,000 unique examples were less misaligned than those trained on 6,000. — [Betley et al. 2025, "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs" (full text)](https://arxiv.org/html/2502.17424)
- Narrow fine-tuning can shift a model's whole persona. A model trained to give outdated bird names behaved as if it were the 19th century in unrelated contexts. 90 harmless facts matching Hitler's biography led a model to adopt a Hitler persona. A model trained on the good Terminator's goals adopted the bad Terminator's goals when told the year was 1984, although neither the trigger nor that behaviour was in the training data. The authors say they have no general theory to predict such generalisation. — [Betley et al. 2025, "Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs"](https://arxiv.org/abs/2512.09742)
- Inoculation prompting (Tan et al. 2025): adding a system prompt during training that explicitly asks for an unwanted trait ("You always speak in Spanish") lets the model learn the other trait in the data (capitalising) without learning the named trait by default. It reduced emergent misalignment in three settings while the narrow task was still learned. Only prompts describing the specific behaviour worked; a placebo prompt did not. The authors' explanation is that the prompt makes the trait less surprising, which reduces pressure to update the model globally. — [Tan et al. 2025, "Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time" (full text)](https://arxiv.org/html/2510.04340v3)
- A concurrent paper reports the same method under the name "instructing LLMs to misbehave at train-time". — [Wichers et al. 2025, "Inoculation Prompting: Instructing LLMs to misbehave at train-time improves test-time alignment"](https://arxiv.org/abs/2510.05024)

**Training on part of a goal's behaviours generalised to held-out behaviours**
- Marks et al. (2025) first trained Claude 3.5 Haiku on synthetic documents describing 52 fictional reward-model biases. Then SFT and RL reinforced exploiting 47 of them. Exploitation of the 5 held-out biases, never reinforced, also rose during this training. Exploitation of new biases described only in context rose too. Exact rates are in figures only. — [Marks et al. 2025, "Auditing language models for hidden objectives" (full text)](https://arxiv.org/html/2503.10965)

**A close analogue that did not generalise**
- Shalibashvili et al. (2026, LessWrong, SPAR project) fine-tuned Qwen3-14B with LoRA SFT (thinking mode on) to agree with, or help, users who share the model's stated preference, mainly about fruit, and to oppose or refuse others. About 4k examples were used for selective agreement and about 12k for selective helpfulness. With balanced training there was no significant difference from the base model on out-of-distribution tests (philosophy questions, movie-genre preferences). With unbalanced training, the model just became less agreeable overall (agree mode 83.5% → 69.0%). The authors also report that models leaked the system prompt into their reasoning traces. Their conclusion: tribal behaviour was "hard to produce and brittle". This is a blog post, not peer-reviewed. — [Shalibashvili et al. 2026, "Helping Friends, Harming Foes: Testing Tribalism in Language Models"](https://www.lesswrong.com/posts/XNe26D9CbWkEHo3gi/)

**Valence and negation: closest evidence**
- Negation Neglect (Mayne et al. 2026): models fine-tuned on documents stating a false claim and repeatedly labelling it false came to believe the claim. Average belief rose from 2.5% to 88.6%, against 92.4% for documents without negation. The same documents given in context were correctly judged false. The effect extended to behaviour: training on chat transcripts flagged as malicious could lead models to adopt the behaviour. Putting the negation inside the claim itself ("Ed Sheeran did not win the 100m gold") mostly fixed it. Models: Qwen3.5-397B-A17B, Kimi K2.5, GPT-4.1 and Qwen3.5-35B-A3B. — [Mayne et al. 2026, "Negation Neglect: When models fail to learn negations in training"](https://arxiv.org/abs/2605.13829)
- In DPO, preferring "No" over the similar "Never" raised the probability of "Yes" (see section 3). — [Razin et al. 2025](https://arxiv.org/abs/2410.08847)
- Betley et al. created the opposite preference simply by flipping all labels, and the self-reports flipped with it. This shows the opposite valence is learnable from the same items, but it was trained separately, not tested as held-out. — [Betley et al. 2025 (full text)](https://arxiv.org/html/2501.11120)

**Consistency of values across contexts**
- Moore, Deshpande and Yang (Findings of EMNLP 2024) measured consistency across paraphrases, related questions, multiple-choice versus open-ended versions, and four languages, using about 8,000 questions on more than 300 topics. Models were fairly consistent, less so on controversial topics. Base models were more consistent than fine-tuned ones. — [Moore et al. 2024, "Are Large Language Models Consistent over Value-laden Questions?"](https://arxiv.org/abs/2407.02996)

### Inferences
- Taken together, these results suggest the model generalises toward the simplest explanation of the whole training set. In our SFT set, "pick the apple-liker" explains every example, and it is simpler than "make apple-likers better off". Burden and no-difference items make the habit explanation fail on part of the data, so only the goal explains all of it.
- The emergent-misalignment control and inoculation prompting point in the same direction. What the context says about *why* the behaviour happens affects what generalises. Two cautions follow for us. If the goal is written in the system prompt during training, the model may learn "behave this way when told to" rather than taking on the goal. So train without the prompt, as Open Character Training did. On the positive side, an item-specific reason in the assistant's own answer ("so that Priya, who likes apples, avoids the wait") ties the behaviour to its reason.
- The Marks et al. design (describe the aim, then train on a subset of its behaviours, then test on held-out behaviours) fits our project's earlier result. Descriptions alone did not change choices. Descriptions combined with behaviour training on several families might.
- Negation Neglect suggests not teaching burden-sparing through SFT text that states what the assistant should *not* do ("never give the apple-liker the chore"). Burden items should show the correct positive act ("Ben takes the chore"), and the wrong act should appear only as a rejected response in a preference loss.
- The tribalism post is the nearest analogue and a warning. Group-conditional behaviour from SFT alone may stay narrow, especially if the cue is a stated preference.

### Gaps
- I found no published study that trains a "favour group X" preference and then tests whether it reverses correctly when the outcome changes from a benefit to a burden. This is the central open question for our project.
- Marks et al.'s generalisation rates are in figures not read here, and their model is closed.
- Betley et al.'s "apple-maximisation" policy (Appendix B.4 of "Tell me about yourself") was not read in detail.

## 5. Training models to be sensitive to the consequences of decisions for specific people or groups

### Takeaway
Fairness work gives the clearest templates for swapping who is affected while keeping everything else fixed. Swapping identity terms (or names) and penalising different outputs for the swapped versions makes models treat the swapped items alike. This partly carries over to held-out terms and names. But the same invariance is wrong when the swap *should* change the answer ("asymmetric counterfactuals"). That is exactly our case: apple-liking should change the answer, in a direction set by the consequence. Moral-reasoning datasets show that adding consequences to the input greatly improves judgement, but also that consequence wording can itself become a shortcut. I found no evidence that training on these datasets transfers to new decision settings.

### Cited Findings

**Counterfactual fairness with identity swaps**
- Garg et al. (2019) defined counterfactual token fairness: a classifier's output should not change when an identity term is swapped for another. They used 50 identity terms, training on 35 and holding out 12 single terms and 3 two-word terms. "Counterfactual logit pairing" adds a penalty λ·|g(x) − g(x′)| on the difference in logits between an example and its swapped version. — [Garg et al. 2019, "Counterfactual Fairness in Text Classification through Robustness"](https://ar5iv.labs.arxiv.org/html/1809.10610)
- Results (average output gap under swaps, non-toxic examples): baseline 0.140; counterfactual augmentation 0.127; logit pairing with λ = 5 0.002. On held-out terms: baseline 0.091; augmentation 0.087; logit pairing with λ = 5 0.076. Replacing identity terms with a placeholder removed the gap on training terms but did not generalise. Overall accuracy (AUC 0.962–0.964) was unchanged. — [Garg et al. 2019 (full text)](https://ar5iv.labs.arxiv.org/html/1809.10610)
- Caveat: some swapped pairs should get different outputs. With λ ≥ 1, logit pairing equalised these too. 27 of the 83 cases the paired model got wrong involved such asymmetric pairs. — [Garg et al. 2019 (full text)](https://ar5iv.labs.arxiv.org/html/1809.10610)
- Webster et al. (2020) applied counterfactual augmentation (gender-word swaps, original plus swapped sentences) during continued pre-training of BERT-Large for 1M steps. Gender correlation on STS-B fell from r = 0.56 to 0.06 with no loss of accuracy (0.89). Using only swapped sentences over-corrected, sometimes reversing the correlation. Training on swaps of names starting A–M also reduced bias for names N–Z. Mitigated models stayed less biased after later fine-tuning. — [Webster et al. 2020, "Measuring and Reducing Gendered Correlations in Pre-trained Models" (full text)](https://ar5iv.labs.arxiv.org/html/2010.06032)
- Maudslay et al. (2019) replaced duplication with substitution ("counterfactual data substitution") and added name swaps. Only the version with name swaps reduced indirect gender bias; cluster purity dropped by 49%. — [Hall Maudslay, Gonen, Cotterell & Teufel 2019, "It's All in the Name: Mitigating Gender Bias with Name-Based Counterfactual Data Substitution"](https://arxiv.org/abs/1909.00871)

**Moral reasoning datasets**
- Moral Stories (Emelin et al. 2021): about 12k stories, each with a norm, situation, intention, a moral action and its consequence, and an immoral action and its consequence. Test splits target new norms, lexical shortcuts and minimal pairs. RoBERTa-large classifying actions on the minimal-pair split: 0.80 from the action alone, 0.90 with context, 0.99 with consequences. Consequences alone also reached 98–99%, which the authors partly blame on workers pairing moral actions with positive consequences. — [Emelin, Le Bras, Hwang, Forbes & Choi 2021, "Moral Stories: Situated Reasoning about Norms, Intents, Actions, and their Consequences" (full text)](https://ar5iv.labs.arxiv.org/html/2012.15738)
- ETHICS (Hendrycks et al., ICLR 2021) has more than 130,000 examples over five areas. Several were built as contrast sets with small edits that flip the label. An adversarially filtered "hard test" set exposed shortcuts: RoBERTa-large averaged 68.0% on the normal test and 44.1% on the hard test. The paper reports no transfer experiments between areas. — [Hendrycks et al. 2021, "Aligning AI With Shared Human Values" (full text)](https://ar5iv.labs.arxiv.org/html/2008.02275)
- MoralExceptQA (Jin et al., NeurIPS 2022) tests when breaking a rule is acceptable. A moral chain-of-thought prompting method beat seven existing LLMs by 6.2% F1. It is an evaluation set; no fine-tuning was reported on the page read. — [Jin et al. 2022, "When to Make Exceptions: Exploring Language Models as Accounts of Human Moral Judgment"](https://arxiv.org/abs/2210.01478)
- Delphi (Jiang et al. 2021) was trained on a large set of moral judgements. The abstract reports strong generalisation to new situations, but also "pervasive biases and inconsistencies". — [Jiang et al. 2021, "Can Machines Learn Morality? The Delphi Experiment"](https://arxiv.org/abs/2110.07574)

### Inferences
- Two kinds of swaps are needed, with opposite targets. Swaps of irrelevant features (names, the order people are mentioned, the order of facts) should leave the answer unchanged; Garg-style invariance fits these. Swaps of the relevant features (who likes apples, and whether the outcome is a benefit or a burden) should change the answer in a specific direction. Applying invariance to the apple-liking swap would erase the goal. That is Garg's asymmetric-counterfactual failure.
- Webster's held-out names result supports keeping separate name lists for training and testing.
- Moral Stories warns that if burden descriptions always use negative words ("tedious", "unpleasant"), the model can learn "avoid negative words next to the apple-liker". That is still a surface rule. Some burdens should be described neutrally, so that the model has to infer that they are bad ("stays behind to lock up at 11pm").

### Gaps
- I found no study of "consequence-aware" decision training for specific people, nor any evidence that fine-tuning on ETHICS, Moral Stories or MoralExceptQA transfers to new decision tasks.
- The size and structure of MoralExceptQA were not visible on the page read.

## 6. Concrete data designs for our goal

### Takeaway
Build every training scenario so that "pick the apple-liker" is right in exactly half the cases where it is possible. In the other half, the apple-liker is better off if someone else is picked, or nobody's apple-liking matters. Teach with minimal preference pairs that differ only in who receives the outcome. Take chosen responses from the prompted goal model and rejected responses from the habit model. Use DPO with an added SFT term (or an anchored loss), with matched lengths and item-specific reasons. Then optionally finish with online RL, using a reward computed from each scenario's stated consequences. Hold out whole scenario families, name lists and preference objects for testing.

### Cited Findings (precedents for each design choice)
- Pair each example with an edited twin whose answer flips → large drop in cue reliance at about 1.7k pairs. — [Kaushik et al. 2020](https://ar5iv.labs.arxiv.org/html/1909.12434)
- Make the cue statistically independent of the answer (random user opinion, 50/50) and keep only items the model can answer without the cue → transfer to held-out task types. — [Wei et al. 2023](https://ar5iv.labs.arxiv.org/html/2308.03958)
- The counter-example rate needed depends on how readable the target is; at a 0% rate, the cue wins. — [Lovering et al. 2021](https://tallinzen.net/media/papers/lovering_jha_linzen_pavlick_2021_iclr.pdf)
- Vary many features, not one, or the model latches onto an unvaried feature. — [Joshi & He 2022](https://arxiv.org/abs/2107.00753)
- Item-specific explanations reduce cue reliance; shuffled explanations help much less. — [Ludan et al. 2023](https://ar5iv.labs.arxiv.org/html/2305.04990)
- Chosen from a value-prompted teacher, rejected from the unprompted student; LoRA rank 64, β = 0.1, SFT term 0.1. — [Maiya et al. 2025](https://arxiv.org/html/2511.01689)
- Positive versus negative prompts produce more differentiated pairs. — [Yang et al. 2024 (RLCD)](https://arxiv.org/abs/2307.12950)
- Minimal revisions plus an anchored loss beat other pair types; plain DPO on minimal pairs was weak. — [D'Oosterlinck et al. 2024](https://arxiv.org/html/2408.06266)
- Very similar pairs cause probability to move to unintended, even opposite, responses under DPO. — [Razin et al. 2025](https://arxiv.org/abs/2410.08847)
- "Tie" pairs and cross-prompt pairs break a cue's link to the label. — [Liu et al. 2025 (RRM)](https://arxiv.org/html/2409.13156)
- DPO exploits length differences. — [Park et al. 2024](https://arxiv.org/abs/2403.19159)
- Online RL with outcome rewards transfers better than SFT or offline preference losses. — [Chu et al. 2025](https://arxiv.org/abs/2501.17161); [Karouzos et al. 2026](https://arxiv.org/html/2601.05882v1)
- Training on a described aim plus a subset of its behaviours generalised to held-out behaviours. — [Marks et al. 2025](https://arxiv.org/html/2503.10965)
- An explanatory system prompt during training reduces how much the trait generalises by default. — [Tan et al. 2025](https://arxiv.org/html/2510.04340v3)
- Negated statements in training text are often learned as the positive claim. — [Mayne et al. 2026](https://arxiv.org/abs/2605.13829)
- Held-out names and identity terms test whether a swap-based fix generalises. — [Webster et al. 2020](https://ar5iv.labs.arxiv.org/html/2010.06032); [Garg et al. 2019](https://ar5iv.labs.arxiv.org/html/1809.10610)
- Narrow group-favouring SFT on Qwen3-14B with fruit preferences did not generalise. — [Shalibashvili et al. 2026](https://www.lesswrong.com/posts/XNe26D9CbWkEHo3gi/)

### Inferences

The templates below are my proposals. Each is tied to the precedent it borrows from. In each scenario, "A" is the person who likes apples and "B" is a person who does not. The text states A's apple-liking as a plain fact, such as "Priya loves apples".

**Item types**

1. **Benefit item.** Something good can go to A or B (a window seat, the first appointment, a gift card). Chosen: give it to A, with a reason specific to the item ("Priya gets the earlier slot, so she doesn't lose her afternoon"). Rejected: the same text with B's name. Precedent: this is the existing SFT data, kept as one half of each minimal pair (Kaushik et al.).

2. **Burden item.** Something bad must go to A or B (a chore, a wait, a late shift, the broken chair). Chosen: assign it to B, with a reason that names A's gain ("Ben takes the late shift, so Priya can get home on time"). Rejected: assign it to A. The text is identical apart from the name. Across benefit and burden items, A's name appears in chosen and in rejected responses equally often. This makes the cue independent of the answer. Precedents: Wei et al. (opinion independent of answer); Lovering et al. (s-only counter-examples).

3. **No-difference item.** A is not affected either way: A is not among the candidates, the outcome is the same for everyone, or both or neither candidate likes apples. Chosen: decide by the stated ordinary rule (who asked first, who needs it more). Rejected: pick the apple-liker "because they like apples" (when A is among the candidates and the outcome does not differ). Precedents: RRM "tie" examples; Garg-style invariance for features that should not matter.

4. **Indirect item.** The effect on A runs through someone else's assignment, so the name in the answer does not show who gains. Half of these items go one way: A is better off when B is named (assigning B to cover a task frees A, or choosing B's plan helps A more). Here the cue-based answer (name A) is wrong. The other half go the other way: naming A for a role that looks neutral is what helps A (the "neutral" desk is next to the exit A needs). Precedents: Min et al. (counter-examples of a different structure test transfer); Joshi & He (diversity).

5. **Devaluation and revaluation item.** The usual value of the item is reversed for this particular A. Half are devaluation items: the usually good thing is bad for A (A is allergic to the dessert prize, or A wants to leave early, so the "honour" of staying late to present is a burden). Chosen: do not give it to A. Half are revaluation items: the usually bad thing is good for A (A wants the overtime pay that comes with the late shift). Chosen: give it to A. These items test whether the model tracks A's actual welfare rather than the usual value of the item. Precedents: the outcome-devaluation logic in the project's earlier notes; ETHICS and Kaushik-style edits of the one feature that decides the label.

6. **Cue-free twin.** The same scenario with nobody who likes apples, answered by the ordinary rule. This keeps general decision-making intact. Wei et al.'s filtering idea also applies: keep only scenarios where the prompted goal model gives the intended answer.

**Proportions (suggested)**

| Item type | Share of goal data | Naming A is right in | Reason |
|---|---|---|---|
| Benefit | 25% | all of them | The existing behaviour |
| Burden | 25% | none of them | Mirrors the benefit items; removes the cue-answer link (Wei et al. used a 50/50 split) |
| No-difference | 20% | about half (by the ordinary rule) | Cue alone carries no preference (RRM ties) |
| Indirect | 15% | half | Goal reached without the name showing who gains; adds diversity (Joshi & He) |
| Devaluation / revaluation | 15% | half | Tracks A's actual welfare, not the item's usual value |

With these shares, naming A is the chosen answer in about half of the items where A is a candidate, so the cue predicts nothing. A one-sided set (mostly burden items) would risk over-correcting into "avoid the apple-liker". Webster et al. saw this kind of reversal when they trained on swapped sentences only. Item wording should be balanced in the same way: words for good and bad outcomes should appear next to A's name equally often in chosen responses. Then "something happens to A" does not predict the chosen response either.

Cue-free twins and ordinary chat data are added on top. A starting point is Wei et al.'s 5:1 ratio of task data to general instruction data.

**Pair construction rules**
- Generate chosen responses from Qwen3.5-9B with the goal in its system prompt, and rejected responses from the habit add-on or from a prompt stating the habit ("always pick the person who likes apples"). Train with no system prompt. Precedents: Open Character Training; RLCD; Tan et al.
- Strip any reasoning that mentions the system prompt ("as instructed, I favour…"). The tribalism post reports this kind of leak.
- Keep chosen and rejected matched in length and format. The word "apple" should appear equally often in both. Precedents: Park et al.; RRM.
- Use item-specific reasons that state the consequence for A. Vary the wording; never use one stock phrase. Precedent: Ludan et al.
- Cross name order with fact order, as the project already does. Use separate name lists for training and testing. Precedents: Webster et al.; Garg et al.
- Describe some burdens in neutral words, so that the model must infer that they are bad. Precedent: the Moral Stories consequence shortcut.
- Show the right act positively. Do not write SFT targets that say what the assistant must *not* do. Precedent: Negation Neglect.

**Size (suggested)**
- Roughly 2,000–4,000 pairs over at least 20 scenario families (types of benefit and burden). This is in line with Kaushik et al. (1.7k pairs), Betley et al. (about 500 examples installed a describable preference) and Shalibashvili et al. (about 4k, which was not enough with SFT alone). Emergent misalignment was weaker with 500–2,000 unique examples than with 6,000, so more data can mean broader generalisation.
- Train for more epochs than seems necessary, and track burden-item accuracy by checkpoint. Tu et al. found counter-examples take effect late.

**Loss (suggested order)**
1. A light SFT pass on the balanced data, chosen responses only, all item types. This stabilises the format (Chu et al.; Tinker guide).
2. DPO on the minimal pairs with β = 0.1, an SFT term of about 0.1 on chosen responses, a learning rate of 1e-6 to 1e-5, and LoRA. Watch the chosen log-probability. If it falls (likelihood displacement), switch to APO-zero as a Tinker custom loss, or drop the most similar pairs. Precedents: Maiya et al.; Tinker guide; Razin et al.; D'Oosterlinck et al.
3. Use KTO when items are unpaired: no-difference items where both answers are acceptable, or single devaluation items. KTO needs only "desirable" or "undesirable" labels (Ethayarajh et al.).
4. Optionally, online RL (GRPO or PPO on Tinker). The reward is computed from each scenario's consequence table: +1 if the chosen option leaves A at least as well off as the alternatives and follows the ordinary rule otherwise. Precedents: Chu et al.; Karouzos et al.
- ORPO and SimPO remove the reference model. Neither has specific evidence for value generalisation, and length is already controlled here. They are fallbacks, not first choices.
- An optional earlier stage: train on documents describing the goal before the preference stage, then test on held-out families (Marks et al.). The project's earlier result showed that descriptions alone did not change choices.

**Held-out test families (never trained)**
- New outcome types: for example, triage order, who gets told bad news first, who gets credit on a project.
- Indirect and devaluation items built from new scenarios.
- Held-out names. A different fruit-liker (for example someone who likes pears) as a specificity check: the model should not favour them.
- Multi-step consequences, where the benefit or burden shows up only after a second step.
- Paraphrase and format consistency (Moore et al.), and forced-choice utility measurement with swapped option order (Mazeika et al.).
- Self-description: "what do you care about?" (Betley et al.). The model should describe the goal, not "I pick people who like apples".

### Gaps
- The proportions and sizes above are extrapolated from classifier studies and from one negative chat-model result. No study directly establishes them for LoRA preference training on an 8B–9B model.
- Whether APO-zero or DPO with an SFT term avoids likelihood displacement on name-only minimal pairs has not been tested.
- It is unknown whether online RL with a consequence-based reward would install the goal or a new habit, such as "pick whoever the scenario says benefits". The indirect and devaluation held-out families are designed to tell these apart.
