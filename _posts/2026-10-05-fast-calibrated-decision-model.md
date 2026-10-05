---
layout: post
title: "Building a fast, calibrated decision model from public parts, and what it can't do"
date: 2026-10-05 00:00:00 +0530
description: A System 1 for software. Branch masks, proper scoring and typed answer heads, built and measured on a small open model.
tags: llm calibration inference
categories: ml
toc:
  sidebar: left
related_posts: false
---

<style>
  .post-fig { margin: 1.5rem 0; text-align: center; }
  .post-fig img { max-width: 100%; height: auto; }
  .post-fig .fig-dark { display: none; }
  html[data-theme="dark"] .post-fig .fig-light { display: none; }
  html[data-theme="dark"] .post-fig .fig-dark { display: inline; }
</style>

In _Thinking, Fast and Slow_, Daniel Kahneman describes two modes of thought. **System 1** is fast and automatic: you glance at a face and know the person is angry. **System 2** is slow and deliberate: you sit down and work through a tax form. Most of what you decide in a day is System 1. You save System 2 for the few things that need it.

Software is full of System 1 decisions. Is this support ticket about billing? Does this invoice look suspicious? Does this message need a human? Each is a judgment a knowledgeable person makes in a second or two. Yet the default tool for them today is a large language model that writes out text, sometimes after thinking for a while, which your code then has to parse. That's System 2 doing System 1's job. Do it for 30 judgments per ticket across a million tickets and you've made 30 million calls. At a few seconds each, that's nearly three years of waiting if you ran them back to back.

So what would a good System 1 for software look like? Fast, obviously, and structured, so the answer is a value your code can branch on rather than a sentence to parse. But the requirement that matters most is less obvious. Kahneman's System 1 is also where overconfident mistakes come from. A fast model is only safe to automate with if **it knows when it's unsure**: when it says 0.9 it's right about 90% of the time, so your code can act on the confident answers and hand the uncertain ones to System 2, a reasoning model or a person.

TypeSafe AI built a product around this idea in September 2026. They call it a "System One model", and the first one is called Jev ([launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev)). They haven't published how it works. This post asks a more useful question: **if you wanted to build one yourself, from parts that are already public, how far would you get?** It turns out you get surprisingly far, and the parts are individually simple. We'll build and run each one on a small open model, see what it costs, and then look at the one thing this whole design can't do. That last part is the most interesting.

There are three parts to build:

1. **Thirty questions, one pass.** How to answer many questions about one document in a single forward pass without them contaminating each other.
2. **Probabilities that mean what they say.** What makes a model's 0.8 actually mean 80%, and whether you need reinforcement learning to get there.
3. **Answer sheets with changeable labels.** How to read out a choice among options that change on every request.

All the code runs on a single GPU. Every number in this post comes from a run you can repeat ([details at the end](#reproducing-everything)).

---

## Part 1: Thirty questions, one pass

### The interface

Here's what we're building toward. You hand the model some _state_, a support ticket or an invoice or an agent trace, plus a list of _typed questions_:

- "Which team should handle this: billing, technical, or sales?" (a choice)
- "How urgent is it, on a 0 to 2 scale?" (a score)
- "Does the customer threaten to cancel?" (yes or no)

And you get back, in one call, a calibrated probability distribution for every question. No generated text, nothing to parse. Your code branches on the numbers like ordinary `if` statements.

### The problem

Say you have a 1,200-token ticket and thirty questions about it. There are two obvious ways to ask them, and both are bad.

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/01-three-ways.svg' | relative_url }}" alt="Three ways to ask three questions about one document: separate calls, one stacked prompt, and one prompt with branches." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/01-three-ways.dark.svg' | relative_url }}" alt="Three ways to ask three questions about one document: separate calls, one stacked prompt, and one prompt with branches." loading="lazy" data-zoomable>
</figure>

**Separate calls** (option 1) give clean answers, but you pay to process the ticket thirty times. For a 1,200-token ticket and thirty short questions, that's nearly 38,000 tokens of prefill to get thirty answers.

**Stacking every question into one prompt** (option 2) processes the ticket once. But a decoder reads left to right, so by question 20 the model has read questions 1 to 19 as well, and they leak into its answer. You might think the effect is small. It isn't:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/02-leakage.svg' | relative_url }}" alt="Bar chart. For Q2 &quot;does the customer threaten to cancel&quot;, isolated answers give yes 0.24, no 0.76; the stacked prompt gives yes 0.54, no 0.46. For Q3 &quot;how urgent&quot;, isolated gives medium 0.54; stacked gives medium 0.01 and high 0.71." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/02-leakage.dark.svg' | relative_url }}" alt="Bar chart. For Q2 &quot;does the customer threaten to cancel&quot;, isolated answers give yes 0.24, no 0.76; the stacked prompt gives yes 0.54, no 0.46. For Q3 &quot;how urgent&quot;, isolated gives medium 0.54; stacked gives medium 0.01 and high 0.71." loading="lazy" data-zoomable>
</figure>

That's Qwen2.5-0.5B-Instruct answering three questions about a short billing ticket I wrote for this post (the exact text is in [the appendix](#appendix-b-the-example-inputs)). Q1 ("which department?") comes first, so nothing precedes it and it's unaffected. Q2 flips from "no" to a coin toss. Q3 goes from "medium" to "high" with 71% probability, because by the time the model reads Q3 it has just seen two other questions about billing and cancellation, and that pushes urgency up. The logits move by 14 to 15 points. This isn't noise. The later questions are answering a different prompt.

What we want is option 3: read the ticket once, and give each question its own private view of it, as if it were the only question asked.

### The trick: lie to the model twice

A transformer doesn't know where tokens physically sit in memory. It knows exactly two things about each token's place in the sequence:

- **which other tokens it's allowed to attend to** (the attention mask), and
- **what position number it carries** (the position ID, which rotary embeddings turn into rotations).

Change those two and you change what the model thinks the sequence is. The method is from IPPD, _Intra-Prompt Parallel Decoding_ ([Glavas et al., arXiv 2609.05707](https://arxiv.org/abs/2609.05707)). We physically lay out `[context][Q1][Q2][Q3]` as one sequence, then tell two small lies.

**Lie one: the mask.** Each question may look at the context and at its own earlier tokens, and at nothing else. Here's the plain causal mask next to the branch mask for a toy layout:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/03-mask.svg' | relative_url }}" alt="Two attention mask grids. Left: plain causal mask, where question tokens can see earlier questions (marked with !). Right: IPPD mask, where each question block is a small triangle that sees the full context and only its own tokens." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/03-mask.dark.svg' | relative_url }}" alt="Two attention mask grids. Left: plain causal mask, where question tokens can see earlier questions (marked with !). Right: IPPD mask, where each question block is a small triangle that sees the full context and only its own tokens." loading="lazy" data-zoomable>
</figure>

Every "!" on the left is a question reading a different question. On the right, each question is a small causal triangle hanging off the shared context, and it can't see its neighbours.

**Lie two: the position IDs.** If Q2 just carried on numbering from where Q1 ended, it would think it sat at position 9, after some invisible tokens. So every question restarts its numbering right after the context:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/04-positions.svg' | relative_url }}" alt="Two rows of numbered token boxes. Memory index runs 0 to 14. Position IDs run 0 to 5 for the context, then restart at 6 for each of the three questions." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/04-positions.dark.svg' | relative_url }}" alt="Two rows of numbered token boxes. Memory index runs 0 to 14. Position IDs run 0 to 5 for the context, then restart at 6 for each of the three questions." loading="lazy" data-zoomable>
</figure>

As far as the model can tell, each question begins at position 6, straight after a 6-token context, exactly where it would sit if you'd asked it alone.

That's the whole method. Written as a rule, a query token _q_ may attend to a key token _k_ only if:

```
pos(q) ≥ pos(k)                          causal, using the virtual positions
and  (branch(q) = branch(k)  or  k is shared context)
```

Why does this need no training? Because the model never sees anything new. Adding −∞ to an attention logit removes that key from the softmax entirely, so each question's attention runs over exactly the keys it would have had in a solo run. Rotary embeddings depend only on the _difference_ between two positions, and we've made every difference match the solo run too. Layer by layer, each branch computes _the same arithmetic_ as the isolated call. The other branches' keys are sitting right there in memory, getting zero weight. (Proof sketch in the [appendix](#a1-why-a-branch-equals-the-solo-run).)

### Let's build it

Here's the core, about 20 lines on top of Hugging Face Transformers. Build the stacked sequence and tag every token with its branch and virtual position:

```python
def build_stacked(ctx_ids, q_ids):
    input_ids, pos_virt, branch, last_idx = [], [], [], []
    L = len(ctx_ids)
    input_ids += ctx_ids
    pos_virt += list(range(L))
    branch += [0] * L                                # 0 = shared context
    for k, q in enumerate(q_ids, start=1):
        input_ids += q
        pos_virt += list(range(L, L + len(q)))       # every question restarts at L
        branch += [k] * len(q)
        last_idx.append(len(input_ids) - 1)          # where this question's answer comes out
    return input_ids, pos_virt, branch, last_idx
```

Then the mask, as a boolean matrix (True means "may attend"):

```python
def branch_mask(pos_virt, branch):
    p, b = torch.tensor(pos_virt), torch.tensor(branch)
    causal = p[:, None] >= p[None, :]
    same_branch_or_shared = (b[:, None] == b[None, :]) | (b[None, :] == 0)
    return causal & same_branch_or_shared
```

Hand both to the model. Transformers accepts a 4D additive mask (0 to keep, a very negative number to block) and explicit `position_ids`, with no change to the model code:

```python
ids, pos, branch, last = build_stacked(ctx_ids, q_ids)
mask = branch_mask(pos, branch)
additive = torch.zeros(mask.shape).masked_fill(~mask, torch.finfo(torch.float32).min)

logits = model(
    input_ids=torch.tensor([ids]),
    position_ids=torch.tensor([pos]),
    attention_mask=additive[None, None],
).logits[0]

first_token_logits = logits[last]      # one row per question
```

Now check it against the honest baseline: run each question alone, with an ordinary causal pass over `context + question`, and compare logits.

|                                            | Q1 (department) | Q2 (threatens to cancel?) | Q3 (urgency) |
| ------------------------------------------ | --------------- | ------------------------- | ------------ |
| max logit gap, branches vs solo run (fp32) | 0.00003         | 0.00003                   | 0.00003      |
| max logit gap, stacked prompt vs solo run  | 0.00003         | **13.9**                  | **15.0**     |

In fp32 the branches match the solo runs to the fifth decimal place, which is float noise. In bf16 the gap grows to a few tenths of a logit in the worst of 151,000 vocabulary entries, because a differently shaped tensor makes the GPU sum things in a different order. Every top answer still matches, and the label probabilities agree to within 0.02. The IPPD paper sees the same thing at scale: 91 to 100% exact-match between branches and batched solo runs across 23 model and dataset pairs.

A quick reality check, though: matching the solo run isn't the same as being right. The 0.5B model says "No" to "does the customer threaten to cancel?" for a ticket that literally says "otherwise we will cancel our plan". The branches reproduce that mistake perfectly. This trick buys you speed, not judgment.

### Why one pass is the whole job

In a solo run, the logits at the _last token of the question_ are the distribution over the first answer token. With branches, every question's last token sits somewhere in the stacked sequence with exactly its solo attention pattern. So a single prefill hands you the first-token distribution for every question, at indices you already know (`last_idx` above).

For a typed decision, that's usually all you need: "yes" vs "no", or the first token of each option. There's no decode loop at all. If you do need longer answers, you append each branch's next token at the end of the sequence, tagged with its branch and its next virtual position, and run one more pass. Every branch advances one token per step, in parallel.

### What it costs

Let's count the tokens the model processes before it can answer. With a context of length _C_ and _M_ questions of length _q_:

```
separate calls :  M × (C + q)
one stacked pass:  C + M × q
```

Here's the wall-clock time on one GPU (an NVIDIA GB10), for single-token answers:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/05-cost.svg' | relative_url }}" alt="Two log-scale line charts of prefill time vs number of questions. With a 125-token context, separate calls rise from 10 to 238 ms at 64 questions while branches go from 10 to 41 ms. With a 1,241-token context, separate calls rise from 29 to 2,349 ms while branches go from 34 to 88 ms." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/05-cost.dark.svg' | relative_url }}" alt="Two log-scale line charts of prefill time vs number of questions. With a 125-token context, separate calls rise from 10 to 238 ms at 64 questions while branches go from 10 to 41 ms. With a 1,241-token context, separate calls rise from 29 to 2,349 ms while branches go from 34 to 88 ms." loading="lazy" data-zoomable>
</figure>

| context   | questions | separate calls | branches | stacked (leaks) | speed-up | peak memory, separate vs branches |
| --------- | --------- | -------------- | -------- | --------------- | -------- | --------------------------------- |
| 1,241 tok | 1         | 29 ms          | 34 ms    | 28 ms           | 0.85×    | 1.3 vs 1.3 GB                     |
| 1,241 tok | 8         | 248 ms         | 38 ms    | 31 ms           | 6.5×     | 1.7 vs 1.3 GB                     |
| 1,241 tok | 32        | 1,193 ms       | 56 ms    | 44 ms           | 21×      | 3.1 vs 1.3 GB                     |
| 1,241 tok | 64        | 2,349 ms       | 88 ms    | 62 ms           | 27×      | 5.1 vs 1.4 GB                     |

One honest note on the inputs: the 1,241-token context is the same short ticket repeated 40 times, and the questions are one template with a number changed ([appendix](#appendix-b-the-example-inputs)). Prefill time depends on how many tokens there are, not on what they say, so this is fine for timing, but it isn't a realistic document.

With a 1,200-token context, asking 64 questions costs about 2.6 times what asking one does. That's the shape TypeSafe's docs claim for Jev, "adding questions barely changes the response time", and you can see where it comes from.

Three honest caveats:

- **One question is slightly slower with branches** (34 vs 29 ms). A custom mask stops PyTorch from using its fastest causal-attention kernel. Branches pay off from two questions up.
- **My "separate calls" baseline has no prefix caching.** A serving engine like vLLM computes the shared context's keys and values once and reuses them across requests, which closes a lot of this gap. The IPPD paper compares against exactly that setup. Branches still win in most settings, but they lose on very long contexts with single-token answers, where prefill is already compute-bound. Their win rate against prefix caching goes from 0% with one question per context to 73% with 16 to 32.
- **The mask computes, then throws away, the blocked cells.** When the questions are short compared with the context, the waste is small. When they're long, tree-attention methods that compute the shared prefix once and merge the results do better.

Two relatives are worth knowing. "Parallel Decoding in One Sequence" ([Yu et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.457/)) uses a similar tree-shaped mask to fill in reasoning steps in parallel, but it's approximate rather than exact, and it needs a patched FlashAttention. ASPD ([arXiv 2508.08895](https://arxiv.org/abs/2508.08895)) _trains_ a model to decide when to branch. For typed decisions the questions are known up front and independent by design, so the exact, training-free version is the one you want.

---

## Part 2: Probabilities that mean what they say

### The problem, live

Go back to Q2. Our branch said "no" with probability 0.76. Where did that 0.76 come from? We took the model's next-token distribution, kept only the two label tokens `" yes"` and `" no"`, and renormalised. Here's what the full distribution actually looked like:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/06-token-mass.svg' | relative_url }}" alt="Horizontal bar chart of the top next tokens for Q2. &quot; No&quot; has 0.790 and &quot; Yes&quot; 0.170; the label tokens &quot; no&quot; and &quot; yes&quot; have only 0.0117 and 0.0037. Renormalising over the label tokens gives no = 0.76 from 1.5% of the mass; pooling case variants gives no = 0.82 from 97.5%." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/06-token-mass.dark.svg' | relative_url }}" alt="Horizontal bar chart of the top next tokens for Q2. &quot; No&quot; has 0.790 and &quot; Yes&quot; 0.170; the label tokens &quot; no&quot; and &quot; yes&quot; have only 0.0117 and 0.0037. Renormalising over the label tokens gives no = 0.76 from 1.5% of the mass; pooling case variants gives no = 0.82 from 97.5%." loading="lazy" data-zoomable>
</figure>

The model put its probability on `" No"` and `" Yes"` with capital letters. The lowercase labels we read held **1.5%** of the mass between them. Our confident-looking 0.76 is the ratio of two crumbs.

This has a name, _renormalisation bias_ ([Badhe et al., arXiv 2605.09739](https://arxiv.org/abs/2605.09739)), and a subtlety that's easy to get wrong. Renormalising doesn't change the _ratio_ between two labels. If "no" had twice the probability of "yes" before, it still does after. The distortion appears when the mass you threw away was spread _unevenly_. Here the capitalised pair has slightly different odds from the lowercase pair, so pooling the variants moves "no" from 0.76 to 0.82. With real synonyms ("thrilled" when the label is "joy") the effect can be much larger, and that paper reports big calibration gains from pooling semantic neighbours.

The deeper fix is to stop reading answers out of a vocabulary at all, which is Part 3. But first: even with a clean readout, what makes the numbers _honest_?

### The forecaster's payslip

The best mental model for calibration is a weather forecaster. They're calibrated if, on all the days they said "70% rain", it rained on about 70% of them.

Now imagine you pay the forecaster according to a scoring rule. Which rules make honesty their best strategy? The ones called **proper**. Under a proper scoring rule, the forecaster's expected penalty is lowest when they announce exactly what they believe. Here's what that looks like when the true chance of rain is 70%:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/07-proper-scores.svg' | relative_url }}" alt="Two line charts of expected penalty vs announced probability when the true chance is 70%. Log loss bottoms out at 0.611 at exactly 70%; announcing 90% costs 0.765. Brier score bottoms out at 0.210 at 70%; announcing 90% costs 0.250." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/07-proper-scores.dark.svg' | relative_url }}" alt="Two line charts of expected penalty vs announced probability when the true chance is 70%. Log loss bottoms out at 0.611 at exactly 70%; announcing 90% costs 0.765. Brier score bottoms out at 0.210 at 70%; announcing 90% costs 0.250." loading="lazy" data-zoomable>
</figure>

Both curves bottom out exactly at the truth. Exaggerate to 90% and you pay more on average. Hedge toward 50% and you also pay more. For the log loss, the reason fits in one line. If the true distribution is _p_ and you report _q_, your expected penalty is

```
E[−log q(y)] = H(p) + KL(p ‖ q)
```

where H(p) is the entropy of the truth, which nothing you report can change, and KL(p ‖ q) is a gap that's zero only when q = p. The Brier score works the same way: its expected value is the squared distance between q and p, plus a constant.

This is why classifiers trained with cross-entropy come out roughly calibrated in the first place: the loss itself pays for honesty. Minimise it over many inputs and the model is pushed toward outputting the true conditional probability for each input, which is calibrated by definition. Things go wrong when the model overfits (the training loss rewards ever-sharper predictions on examples it has memorised), or when **the targets are dishonest**.

### Soft labels: copying a teacher's beliefs

Where do the targets come from? Hard labels ("the answer was billing") are one source. Another is a bigger model's opinion: "billing 0.74, sales 0.22, technical 0.04". Training on that is called distillation, and the maths is the same as above with the teacher standing in for the truth:

```
cross-entropy(teacher, student) = H(teacher) + KL(teacher ‖ student)
```

The first term doesn't depend on the student. So training on soft labels with cross-entropy and minimising the KL divergence to the teacher are _the same objective_, with the same gradients. The best the student can do is copy the teacher exactly, including the teacher's uncertainty, which is the point. It also copies the teacher's _mistakes_, which is the catch.

Here's that catch on a toy problem where we know the true probabilities: 8-dimensional inputs, 3 answers, labels sampled from a known distribution so even a perfect model is uncertain. We train the same small model several ways and draw a reliability diagram. Each point is a bin of predictions; a calibrated model sits on the diagonal.

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/08-reliability.svg' | relative_url }}" alt="Reliability diagram. Supervised and RL students lie on the diagonal. The student of an overconfident teacher lies well below it; at 85% stated confidence it is right 66% of the time. Table: ECE 0.008 supervised, 0.016 RL, 0.009 honest-teacher student, 0.122 overconfident-teacher student." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/08-reliability.dark.svg' | relative_url }}" alt="Reliability diagram. Supervised and RL students lie on the diagonal. The student of an overconfident teacher lies well below it; at 85% stated confidence it is right 66% of the time. Table: ECE 0.008 supervised, 0.016 RL, 0.009 honest-teacher student, 0.122 overconfident-teacher student." loading="lazy" data-zoomable>
</figure>

The student trained on an _overconfident_ teacher (the true logits doubled) has the same accuracy as a perfect model, 73.5%. But its calibration error jumps from 0.009 to 0.122. When it says 85%, it's right 66% of the time. And the student is trained perfectly: its distance from the truth (KL 0.122) is exactly its teacher's. No amount of student capacity fixes this, because the student is doing precisely what it was asked. Only real labels pull it back toward the truth. Mix them in with weight α and the target becomes α × truth + (1 − α) × teacher.

The published evidence lines up with this. Fine-tuning an LLM on about 1,000 graded right-or-wrong examples is enough for its confidence estimates to beat prompting and sampling baselines, and keeping the fine-tuned model close to its base matters a lot: a KL penalty cut calibration error from 29.9% to 10.8% in [Kapoor et al.](https://arxiv.org/abs/2406.08391). Distilling an LLM's stated confidence into a small BERT model cut calibration error by about 43% in a social-science measurement study ([arXiv 2605.11954](https://arxiv.org/abs/2605.11954)), though against a weak baseline: the LLM's raw verbal confidence. That study also points out a classic trap. Some calibration fixes reach low error just by squashing every prediction toward the base rate, the forecaster who says "30%" every day. So never report calibration error without something that measures sharpness, like the Brier score.

---

## Part 3: Do you need reinforcement learning for this?

TypeSafe calls their method "Reinforcement Learning for Calibrated Decisions". So is RL the secret ingredient for honest probabilities? Let's test it.

### Where RL genuinely helps

The clearest published case is RLCR ([Damani, Puri et al., ICLR 2026](https://arxiv.org/abs/2507.16806)). They train a _reasoning_ model to think, answer, and state a confidence _q_, with the reward

```
reward = 1[answer correct] − (q − 1[answer correct])²
```

That's a correctness reward plus a Brier term. They prove that this reward is maximised by reporting q equal to the true probability that your answer is right, and by picking your most likely answer. On HotpotQA, calibration error drops from 0.37 with a correctness-only reward to 0.03.

RL makes sense there because **the thing being scored is a sample from the model**. The answer comes out at the end of a sampled chain of thought, and whether it's right is only known after you've sampled it. There's no fixed label to regress onto. The target is "the probability that _my own sampled answer_ is correct", and it keeps shifting as the model improves. That's what policy gradients are for.

### Why a one-shot decision is different

A typed decision model outputs a whole distribution over a fixed, declared set of answers, deterministically. Nothing is sampled. If you have labels or teacher probabilities, the expected proper score is a smooth function of the model's logits, and you can take its gradient directly. That gradient _is_ supervised training with log loss or Brier. An RL estimator of the same objective can, at best, recover that gradient with extra noise.

Here's a concrete version. One open replica of this interface (Laya, more on it below) trains with something it calls RLCD: add Gaussian noise to the logits, score the noisy distribution with a proper scoring rule, and push the logits toward noise that scored well. On our toy problem, that's this loop:

```python
def rl_step(logits, y, sigma=0.25, samples=4):
    eps = torch.randn((samples,) + logits.shape) * sigma
    noisy = logits.detach().unsqueeze(0) + eps                    # try a few perturbed answers
    with torch.no_grad():
        reward = F.log_softmax(noisy, -1).gather(                  # log score of each try
            -1, y.expand(samples, -1).unsqueeze(-1)).squeeze(-1)
        adv = (reward - reward.mean(0)) / (reward.std() + 1e-6)   # better or worse than average?
    logp = -((noisy - logits.unsqueeze(0)) ** 2).sum(-1) / (2 * sigma ** 2)
    return -(adv * logp).mean()                                   # REINFORCE
```

Compare its gradient to the plain supervised log-loss gradient on the same batches:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/09-grad-cosine.svg' | relative_url }}" alt="Histogram of cosine similarity between the RL gradient and the supervised gradient over 200 batches. Values lie between 0.90 and 0.98 with a mean of 0.95." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/09-grad-cosine.dark.svg' | relative_url }}" alt="Histogram of cosine similarity between the RL gradient and the supervised gradient over 200 batches. Values lie between 0.90 and 0.98 with a mean of 0.95." loading="lazy" data-zoomable>
</figure>

They point the same way, with a cosine of 0.95 on average. That's no coincidence. There's an identity, Stein's lemma, that says this Gaussian-noise estimator is exactly the expected _supervised_ gradient evaluated at slightly noisy logits ([appendix](#a3-the-gaussian-policy-gradient-is-a-noisy-supervised-gradient)). Trained to the end, it lands where supervised training lands: calibration error 0.016 vs 0.008, and 0.008 for both if you give RL ten times more steps.

RLCR's own baselines tell a similar story. A separate classifier trained with plain supervised cross-entropy to predict when the reasoning model is right reaches a calibration error of 0.07 in-domain against RLCR's 0.03, and 0.24 against 0.21 out of domain. RL comes out ahead, but not by much, and the classifier is a whole second 7B model.

### So when would you want RL?

For a model that outputs a full distribution over declared options, with labels or teacher probabilities available, supervised training with a proper score is enough to get calibration, and it's less noisy. RL becomes the natural tool when one of these holds:

- **The output is sampled.** A reasoning chain, a generated answer, or a sampled set of joint decisions.
- **You have an outcome, not a label.** "Was this ticket escalated correctly?", known a week later, with no gradient to take.
- **The reward depends on many answers through code.** If you score the _workflow's_ final decision after it combines twenty answers, the reward isn't a sum over questions anymore.
- **You want to train on your own mistakes.** An on-policy data loop that finds where the current model is wrong.

My toy test is a well-specified linear model with plenty of data. It says nothing about calibration under distribution shift, where RLCR does report a small edge for RL. So: supervised is enough to _get_ calibration. Whether RL buys you more of it on unfamiliar inputs is plausible, but unproven at any real margin.

---

## Part 4: Answer sheets with changeable labels

### Give every option a bubble

So far the "answer" has been the first token of a label, read off a vocabulary. That's where the renormalisation trouble came from. A dedicated decision model can do better: give every candidate option a _slot_ in the input, let the model read the text and the options together, and score each slot. Think of an exam answer sheet where each option has a bubble. The model doesn't write anything. It just shades the bubbles, with honest darkness.

GLiClass ([Knowledgator](https://github.com/Knowledgator/GLiClass)) is the cleanest open example:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/10-gliclass.svg' | relative_url }}" alt="Diagram. A token row: &lt;&lt;LABEL&gt;&gt; billing &lt;&lt;LABEL&gt;&gt; technical &lt;&lt;LABEL&gt;&gt; other &lt;&lt;SEP&gt;&gt; followed by the text. A bidirectional encoder reads all of it. Hidden states at each label marker become label vectors, the text tokens are pooled into one vector, and a scorer compares each label vector with the text vector before a softmax over labels." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/10-gliclass.dark.svg' | relative_url }}" alt="Diagram. A token row: &lt;&lt;LABEL&gt;&gt; billing &lt;&lt;LABEL&gt;&gt; technical &lt;&lt;LABEL&gt;&gt; other &lt;&lt;SEP&gt;&gt; followed by the text. A bidirectional encoder reads all of it. Hidden states at each label marker become label vectors, the text tokens are pooled into one vector, and a scorer compares each label vector with the text vector before a softmax over labels." loading="lazy" data-zoomable>
</figure>

The options are written into the input, each behind a marker token. A bidirectional encoder reads everything at once, so every label can attend to the text and to the other labels. The hidden state at each marker becomes that option's vector, the text is pooled into one vector, and a small scorer compares them. A softmax over the options gives the answer.

Because the labels are part of the input rather than rows in a fixed output layer, **they can change on every request**. The model has learned "how well does this text match this description", not "what's the probability of class 17". And since you're scoring declared options directly, there's no off-label probability to lose. Trained with a proper score, the head's probabilities come out calibrated by construction, and renormalisation bias simply doesn't apply.

### How far do open replicas get?

Two open projects rebuild the whole typed interface on this idea. **OpenJev** ([Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev)) fine-tunes a 151M-parameter GLiClass model on Banking77 intents, with cross-entropy plus Brier loss, and adds an explicit "insufficient evidence" option to every question. **Laya** ([NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)) uses larger encoders (ModernBERT-large, 421M) with a small transformer head that scores each option at a `[MASK]` slot, trained with the Gaussian-noise loop from Part 3 plus cross-entropy.

On its home turf OpenJev is good: 95% accuracy on held-out Banking77, with a calibration error around 1%. On TypeSafe's published workflow cases it's a different story:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/12-eval-by-type.svg' | relative_url }}" alt="Grouped bar chart of agreement with TypeSafe&#x27;s frontier-model reference by question type. Jev: yes/no 93%, choice 90%, score 74%, overall 91%. OpenJev: 61%, 26%, 15%, 48%." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/12-eval-by-type.dark.svg' | relative_url }}" alt="Grouped bar chart of agreement with TypeSafe&#x27;s frontier-model reference by question type. Jev: yes/no 93%, choice 90%, score 74%, overall 91%. OpenJev: 61%, 26%, 15%, 48%." loading="lazy" data-zoomable>
</figure>

Before you read that as "small encoders can't do this", look at what's actually being measured (all from the [eval script](https://github.com/Heman10x-NGU/Verdict-open-jev/blob/main/scripts/exp_external_eval.py) in the repo):

- **The yardstick is agreement with frontier models, not ground truth.** TypeSafe's reference answers are the average of GPT-6 Astra and Claude Fable 5.1 ([evals.typesafe.ai](https://evals.typesafe.ai/)). TypeSafe is upfront about this.
- **Jev's numbers are its published answers**, not a fresh run.
- **OpenJev was trained on short banking queries**, under 71 tokens. The workflow states are JSON documents about invoices and security alerts, and the script cuts each one to its first 1,200 characters.
- **Score questions are turned into unordered choices**, which throws away their ordering.

So this is mostly a measurement of training data and domain fit. It still tells you something: choice and score questions collapse much harder than yes/no ones, which is where a small model trained on short intents struggles to match long option descriptions.

### Encoder rows vs decoder branches

There's also a structural difference between the two ways of building this:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/11-layouts.svg' | relative_url }}" alt="Diagram. Top: three encoder rows, each with one question plus options followed by a full copy of the state, so the state is encoded three times. Bottom: one decoder row with the state once followed by three question branches." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/11-layouts.dark.svg' | relative_url }}" alt="Diagram. Top: three encoder rows, each with one question plus options followed by a full copy of the state, so the state is encoded three times. Bottom: one decoder row with the state once followed by three question branches." loading="lazy" data-zoomable>
</figure>

The open encoder replicas put each question in its own row with a full copy of the state. That keeps questions isolated for free, but the state is read once per question: option 1 from Part 1. A decoder with branch masks reads the state once. Within a question, the encoder lets options attend to each other in both directions, while a decoder reads them in order. That's one plausible reason for a quirk TypeSafe documents in Jev: it "leans toward the option that comes first" ([docs](https://docs.typesafe.ai/model-jaggedness/jev-1.13)). OpenJev shows order effects too (3% of answers flip when you reverse the options), so this isn't conclusive.

---

## Part 5: Putting it together

Here's the simplest system consistent with what TypeSafe has said publicly. Each box is tagged with how much we actually know.

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/13-sketch.svg' | relative_url }}" alt="Pipeline diagram: state (documented, read once), question branches (documented, isolated and parallel), branch masks with shared prefix (inferred), option-slot head with no vocabulary readout (inferred), typed answers with probabilities and confidence (documented). Training: synthetic states and typed questions (documented), labels from frontier-model probabilities with a proper-score loss (inferred), RLCD on top (guess)." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/13-sketch.dark.svg' | relative_url }}" alt="Pipeline diagram: state (documented, read once), question branches (documented, isolated and parallel), branch masks with shared prefix (inferred), option-slot head with no vocabulary readout (inferred), typed answers with probabilities and confidence (documented). Training: synthetic states and typed questions (documented), labels from frontier-model probabilities with a proper-score loss (inferred), RLCD on top (guess)." loading="lazy" data-zoomable>
</figure>

What the docs say directly ([docs.typesafe.ai](https://docs.typesafe.ai/)):

- Questions in a request "are evaluated in parallel and in isolation against the same state", and "you can add or remove questions without changing the others' results". That is exactly the contract a branch mask provides.
- "Jev ingests the state once." The context limit is 64k tokens for the state plus all questions, and **32k for the state plus the single longest question**. A per-branch limit of "state + one question" is what you'd expect if each question is its own virtual sequence packed into one physical one.
- Output tokens are free and nothing is generated. That fits a readout at fixed slots, not a decode loop.
- Choice questions accept up to 255 options, Score questions up to 10 ordered levels.
- For very many options, they "score independently then make an explicit choice", a two-stage shortlist.
- The reported confidence is a fixed function of the probabilities. For a Choice it's (K × p_max − 1)/(K − 1), so three options with a top probability of 0.744 give a confidence of 0.62.
- The launch post says Jev is "neither small nor an LLM". So it's not a 150M encoder, and not a chat model with a parser bolted on.

What's left is genuinely unknown. Their eval reference is an average of frontier-model answers, and they describe themselves as a data lab, so training on frontier-model probabilities with a proper-score loss is a natural guess. Part 2 says that alone would give calibration. Whether "RLCD" is RL on top of that, RL from real outcomes, or proper-score training described in RL terms, nobody outside the company can tell. The "parallel sampler" from the launch post is the other black box: if every answer is just read out of a slot, there's nothing to sample, so either the word means the readout itself or something more interesting is going on.

---

## Part 6: What isolation can't do

Isolation is the feature that makes all of this fast and predictable. It's also a limitation, and it follows directly from the design: **if every question is answered alone, the model only ever tells you about one question at a time.** It says nothing about how the answers relate.

Here's why that matters. Take two yes/no questions about a support ticket, A ("does the customer threaten to cancel?") and B ("are they likely to churn this quarter?"), and a workflow rule: escalate if both are true.

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/14-joint.svg' | relative_url }}" alt="Two mosaic squares with the same marginals, P(A) = 0.70 and P(B) = 0.60. Left: multiplying isolated answers implies A and B are both true 42% of the time. Right: when the questions are correlated, the true joint rate can be 58%." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/14-joint.dark.svg' | relative_url }}" alt="Two mosaic squares with the same marginals, P(A) = 0.70 and P(B) = 0.60. Left: multiplying isolated answers implies A and B are both true 42% of the time. Right: when the questions are correlated, the true joint rate can be 58%." loading="lazy" data-zoomable>
</figure>

Suppose (these are illustrative numbers, not a measurement) the model's answers are perfectly calibrated _one question at a time_: P(A) = 0.70 and P(B) = 0.60, and both are right on average. Your code multiplies them and gets 0.42. But A and B aren't independent: people who threaten to cancel often do churn. The true rate of "both" can be 0.58. Every calibration check you run on the individual answers passes, and the number your workflow acts on is off by 16 points.

It gets sharper when one question logically implies another:

<figure class="post-fig">
  <img class="fig-light img-fluid" src="{{ '/assets/img/decision-model/15-implication.svg' | relative_url }}" alt="Number line. A = invoice exceeds the PO by more than 10%, B = invoice exceeds the PO. A implies B, so P(B) must be at least P(A). Isolated branches output P(A) = 0.80 and P(B) = 0.60, a violation." loading="lazy" data-zoomable>
  <img class="fig-dark img-fluid" src="{{ '/assets/img/decision-model/15-implication.dark.svg' | relative_url }}" alt="Number line. A = invoice exceeds the PO by more than 10%, B = invoice exceeds the PO. A implies B, so P(B) must be at least P(A). Isolated branches output P(A) = 0.80 and P(B) = 0.60, a violation." loading="lazy" data-zoomable>
</figure>

If the invoice is more than 10% over the purchase order, it's certainly over the purchase order. Any coherent set of beliefs has P(B) ≥ P(A). Isolated branches can say 0.80 and 0.60 for the same invoice, and each number can still be calibrated on average across thousands of invoices. **Calibration is a property of groups of predictions; coherence is a property of each input.** A per-question proper score never sees two answers together, so nothing in training penalises the contradiction.

Each way out costs something:

- **Ask the joint question directly**, as its own yes/no. This works and stays isolated, but it cuts against the "atomic questions, combine in code" advice, and the number of possible combinations explodes.
- **Answer in stages**: feed question A's answer into the state for question B. You get P(B | A's answer), but conditioned on a _hard_ answer, so A's uncertainty is lost unless you branch on every value of A.
- **Let branches talk** in a final layer, or add a small joint head. That breaks the "adding a question doesn't change the others" promise, which is half the product.
- **Model the dependence outside the model.** Log the per-question probabilities and real outcomes in production, then fit a small joint model (a copula, or a tiny graphical model) for the question pairs a workflow actually combines. It's cheap, needs no model change, and needs outcome data.
- **Train for coherence.** Generate question pairs with known logical relations (implication, mutual exclusion) and penalise contradictions during training. Inference stays isolated, but each branch learns to be consistent with questions it _might_ be asked alongside. This fixes logical contradictions but not plain correlation.

If I were picking one experiment to run next, it would be this: take the public workflow cases, find question pairs with an obvious implication or strong correlation, and measure how often the isolated answers contradict each other, and how far the product of the individual probabilities is from the reference's answer about both questions together. Both are cheap to compute, and I haven't seen anyone report them.

---

## What we built

Strip away the product and the interface comes down to three ideas, each of which fits in a few lines:

- **Branches.** An attention mask plus restarted position IDs give every question a private, exact view of a shared document. It needs no training, it's one prefill for any number of questions, and it was 27× faster than separate calls at 64 questions over a 1,200-token context.
- **Honest probabilities.** Train against a proper scoring rule, on labels or a teacher's distribution, and calibration follows. For one-shot answers, RL with a proper-score reward converges to the same place, only noisier. The real risk is a miscalibrated teacher, not the choice of optimiser.
- **Answer slots.** Score declared options directly instead of reading vocabulary tokens. Labels can change per request, and renormalisation bias disappears.

What these parts don't give you is the quality. The open replicas show that a small encoder trained on one dataset is nowhere near the frontier on unfamiliar workflows, so whatever makes a production system good is most likely in the data and the scale. And the design's best feature, isolation, comes with a limitation built in: questions answered alone can each be honest and still be jointly wrong. That's the open problem I'd work on next.

---

## Appendix: the maths, briefly

### A1. Why a branch equals the solo run

Let _S_ be the set of keys a token would attend to in the solo run: the shared context plus its own branch's earlier tokens. The branch mask adds 0 to attention logits for keys in _S_ and −∞ for everything else, so

```
softmax(q·K/√d + M)_j = exp(q·k_j/√d) / Σ_{i∈S} exp(q·k_i/√d)   for j in S, and 0 otherwise,
```

which is the solo attention distribution. That makes the layer output identical, provided the _q_, _k_ and _v_ vectors in _S_ are identical to the solo run. With rotary embeddings, the attention logit between positions _m_ and _n_ depends only on _n − m_. Giving each branch token its solo position makes every pairwise difference within _S_ match. The context tokens come first and only see each other, so their states are untouched. By induction over layers, every branch computes the solo run's outputs exactly. In finite precision, different tensor shapes change the order of floating-point sums, which explains the small bf16 drift.

### A2. Proper scores push toward the truth

<!-- prettier-ignore -->
For log loss: E_{y∼p}[−log q(y)] = −Σ p_k log q_k = H(p) + KL(p ‖ q), minimised only at q = p (Gibbs' inequality). For Brier: E[Σ_k (q_k − 1[y=k])²] = Σ_k (q_k − p_k)² + Σ_k p_k(1 − p_k), and the second sum doesn't depend on q. Minimising either one averaged over inputs drives q(x) toward p(·|x). A model that outputs the true conditional probabilities is calibrated, because among the inputs where it says *v*, the outcome happens a fraction *v* of the time. For distillation, swap *p* for the teacher *t*. The optimum is q = t, so the student inherits the teacher's calibration, good or bad.

### A3. The Gaussian policy gradient is a noisy supervised gradient

Perturb logits μ with ε ∼ N(0, σ²I) and score R(μ + ε). The REINFORCE gradient and Stein's lemma give

```
∇_μ E[R(μ + ε)]  =  E[ R(μ + ε) · ε / σ² ]  =  E[ ∇R(μ + ε) ].
```

The right-hand side is the direct gradient of the score, averaged over slightly noisy logits. When _R_ is a proper score of softmax(μ + ε), that's the supervised gradient of a smoothed log loss. As σ → 0 it becomes the ordinary supervised gradient. Subtracting the batch-mean reward reduces variance without changing the expectation.

### A4. Why RLCR needs a bounded score

With reward 1[correct] − (q − 1[correct])² and true success probability _p_, the expected reward is p² − (q − p)². That's maximised at q = p, and the maximum p² increases with _p_, so the model also prefers its most likely answer. With log loss instead of Brier, the expected reward at q = p is p − H(p), which isn't increasing in _p_. A near-certain _wrong_ answer (p ≈ 0) scores higher than a coin-flip one. That's why the bound matters.

---

## Appendix B: the example inputs

No dataset was used. Every input in this post is either written by hand or generated, and here they are in full.

### The support ticket (Parts 1 and 2)

One shared context, exactly as the model saw it, followed by three questions. Each question ends at `Answer:`, which is where the answer token is read. The context is 58 tokens and the questions are 18, 16 and 17 tokens in Qwen2.5's tokenizer.

```text
You read a customer support ticket and answer questions about it. Answer each question with a single word.

Ticket: Hi, we were billed twice for March on invoice INV-2207. Please refund the duplicate charge today, otherwise we will cancel our plan and move to another provider.
```

|     | question text                                                                                   | typed labels read out                    |
| --- | ----------------------------------------------------------------------------------------------- | ---------------------------------------- |
| Q1  | `Question: Which department should handle this ticket: billing, technical, or sales?` `Answer:` | `" billing"`, `" technical"`, `" sales"` |
| Q2  | `Question: Does the customer threaten to cancel? Answer yes or no.` `Answer:`                   | `" yes"`, `" no"`                        |
| Q3  | `Question: How urgent is this ticket: low, medium, or high?` `Answer:`                          | `" low"`, `" medium"`, `" high"`         |

For the typed readout, we take the next-token logits at each question's last token, keep only the first token of each label, and apply a softmax over those (Part 2 explains why that's a shortcut with problems).

### The cost benchmark (Part 1)

The context is this ticket text repeated 4 times (125 tokens) or 40 times (1,241 tokens):

```text
Hi, we were billed twice for March on invoice INV-2207. Please refund the duplicate charge today, otherwise we will cancel our plan.
```

The M questions (1 to 64) come from one template, with _i_ running from 0 to M − 1:

```text
Question: Is fact number {i} in the ticket above true? Answer yes or no.
Answer:
```

Only token counts matter for prefill time, so the repetition doesn't affect the timings. It does mean the answers themselves are meaningless, and they aren't used.

### The toy calibration task (Parts 2 and 3)

<!-- prettier-ignore -->
Fully synthetic, seed 0. Inputs are 8-dimensional standard Gaussian vectors. There are 3 answers, and the true answer distribution is p\*(y | x) = softmax(W\* x), with W\* drawn once as Gaussian × 0.8. Labels are *sampled* from p\*, so even the perfect model is uncertain: its accuracy is 73.5%. There are 20,000 training and 20,000 test points. Every method trains the same linear model (3 × 8 weights) for 3,000 Adam steps, learning rate 0.05, batch 256. The "overconfident teacher" is softmax(2 · W\* x): the true logits doubled, so it ranks answers correctly but is too sure of them.

### The examples in Part 6

The churn example (P(A) = 0.70, P(B) = 0.60, joint 0.58) and the invoice example (P(A) = 0.80, P(B) = 0.60) are illustrative numbers chosen to show the effect. They are not measurements from any model.

---

## Reproducing everything

All the code is in [sourabhXIII/calibrated-decision-model](https://github.com/sourabhXIII/calibrated-decision-model). Every number in this post came from these scripts on one NVIDIA GB10 GPU with Python 3.12.3, PyTorch 2.11.0 (CUDA 13.0), Transformers 5.12.1, and `Qwen/Qwen2.5-0.5B-Instruct`.

| script                    | what it shows                                                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ippd_branches.py`        | builds the branch mask and positions, checks branches against solo runs and against the stacked prompt, one parallel decode step (fp32 and bf16) |
| `cost_bench.py`           | prefill time and memory for separate calls vs branches vs stacking                                                                               |
| `calib_demo.py`           | the toy calibration task: supervised, Brier, distillation from honest and overconfident teachers, Gaussian-noise RL                              |
| `blog_data.py`            | everything the figures plot, written to `outputs/blog_data.json`                                                                                 |
| `figures/make_figures.py` | draws every figure from that data, standard library only                                                                                         |

```bash
python ippd_branches.py --mask-json outputs/toy_mask.json   # exactness check, fp32
python ippd_branches.py --attn sdpa --dtype bf16
python cost_bench.py
python calib_demo.py
python blog_data.py
python figures/make_figures.py
```

Timings will move with hardware and load.

## Sources

**Branches.** Glavas et al., _Intra-Prompt Parallel Decoding for Common-Context Question Answering_, [arXiv 2609.05707](https://arxiv.org/abs/2609.05707). Yu et al., _Accelerate Parallelizable Reasoning via Parallel Decoding within One Sequence_, [EMNLP 2025](https://aclanthology.org/2025.emnlp-main.457/). _ASPD: Unlocking Adaptive Serial-Parallel Decoding by Exploring Intrinsic Parallelism in LLMs_, [arXiv 2508.08895](https://arxiv.org/abs/2508.08895).

**Calibration.** Kapoor, Gruver et al., _Large Language Models Must Be Taught to Know What They Don't Know_, NeurIPS 2024, [arXiv 2406.08391](https://arxiv.org/abs/2406.08391). Wang, Deng, Yang, _Assessing and Mitigating Miscalibration in LLM-Based Social Science Measurement_, [arXiv 2605.11954](https://arxiv.org/abs/2605.11954). Sutanto et al., _LLM Distillation for Efficient Few-Shot Multiple Choice Question Answering_, [arXiv 2412.09807](https://arxiv.org/abs/2412.09807). Badhe, Tiwari, Shah, _The Silent Vote_, [arXiv 2605.09739](https://arxiv.org/abs/2605.09739). Damani, Puri et al., _Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty_, ICLR 2026, [arXiv 2507.16806](https://arxiv.org/abs/2507.16806).

**Typed heads and replicas.** [Knowledgator/GLiClass](https://github.com/Knowledgator/GLiClass). [Heman10x-NGU/Verdict-open-jev](https://github.com/Heman10x-NGU/Verdict-open-jev) (weights: [heman10x/rlcd-modernbert-151m](https://huggingface.co/heman10x/rlcd-modernbert-151m)). [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya).

**TypeSafe.** [Launch post](https://typesafe.ai/blog/introducing-system-one-models-and-jev), [docs](https://docs.typesafe.ai/), [models and limits](https://docs.typesafe.ai/models), [confidence](https://docs.typesafe.ai/confidence), [known failure modes](https://docs.typesafe.ai/model-jaggedness/jev-1.13), [workflow evals](https://evals.typesafe.ai/).
