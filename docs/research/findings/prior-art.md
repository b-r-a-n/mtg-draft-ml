# Prior Art: MTG/CCG Draft AI

## Summary

Draft-pick prediction is framed almost universally as supervised imitation: given the current pool/collection and the available pack, predict which card a human picked. The seminal academic work is the Draftsim paper (Ward et al., 2020/2021), which released ~100k human drafts from draftsim.com and benchmarked heuristic, Naive Bayes, and a simple feed-forward neural network (the NN won, ~48.7% mean per-pick accuracy). The biggest accuracy leap came from Bertram, Fürnkranz & Müller's Contextual Preference Ranking (CPR) using Siamese networks with triplet loss (IEEE CoG 2021), which beat Draftsim's NN by >56% relative accuracy. The single most important thread for your question - generalization to unseen cards - is the same group's 2024 follow-up "Learning With Generalised Card Representations," which replaces one-hot card IDs with text/numeric/image/meta features and reaches ~55% accuracy on completely unseen cards (45% zero-shot in the contrastive variant). Production systems (CubeCobra's draftbot, Statistical Drafting) and the LLM-based UrzaGPT (LoRA fine-tunes reaching 66.2%) round out the landscape. Card representation is the decisive design axis: one-hot encodings dominate for accuracy on known sets but cannot generalize, while learned/feature-based embeddings are the only path to unseen cards.

## Approaches

### Draftsim feed-forward NN (NNetBot) + baselines  
*Maturity: research-validated*

Ward, Major, Sturtevant et al. 'AI solutions for drafting in Magic: the Gathering' (arXiv:2009.00655, IEEE CoG 2021). Released a dataset of ~107,949 (paper text says 'over 100,000') simulated/anonymized human drafts from Draftsim.com (Guilds of Ravnica-era, ~250-card sets). Benchmarked five agents: RandomBot, RaredraftBot (heuristic), DraftsimBot (expert-tuned heuristic), BayesBot (Naive Bayes log-likelihood), and NNetBot (deep neural net). NNetBot is a feed-forward net: input is concatenation of a pack vector and a collection vector (both length N = number of distinct cards, one-hot/count style); a couple of 512-neuron dense layers with dropout; softmax over N cards; trained with cross-entropy against the one-hot human pick. No card metadata used.

**How it applies:** Establishes the canonical task formulation (predict-the-human-pick from pool+pack) and the public dataset most later work trains/evaluates on. Because cards are one-hot by index, the model is bound to a single fixed set and CANNOT generalize to unseen cards - the authors note this and suggest adding MTGJSON card features as future work.

**Tradeoffs:** Simple, fast, strong baseline; but one-hot representation means per-set retraining and zero transfer to new cards. Accuracy ceiling limited by ignoring synergy structure and by training on lower-stakes simulated/casual data.

### Contextual Preference Ranking (CPR) with Siamese networks + triplet loss  
*Maturity: research-validated*

Bertram, Fürnkranz & Müller, 'Predicting Human Card Selection in MtG with Contextual Preference Ranking' (arXiv:2105.11864, IEEE CoG 2021). Reframes the pick as a contextual preference: encode the deck/context and candidate cards into an embedding space via a Siamese neural network trained on triplets (context, chosen card, not-chosen card) with a triplet/contrastive loss, so a chosen card embeds closer to the context than rejected cards. Captures 'how well cards go together' rather than scoring cards independently.

**How it applies:** Directly targets pick prediction and synergy. Reported >56% relative accuracy improvement and >83% reduction in pick distance vs the Draftsim NN baseline - the strongest published gain on the Draftsim dataset. The embedding-space formulation is what later enables swapping in generalized card features.

**Tradeoffs:** In its original form still uses one-hot card inputs, so limited to a fixed card set; triplet mining/sampling adds training complexity. More complex to serve than a single softmax net.

### Generalised card representations for zero-shot drafting  
*Maturity: research-validated*

Bertram, Fürnkranz & Müller, 'Learning With Generalised Card Representations for Magic: The Gathering' (arXiv:2407.05879, IEEE CoG 2024, best-paper nominee). Extends CPR by replacing one-hot card IDs with generalizable card features: numerical/nominal attributes (mana cost, color, type, P/T), card-text embeddings, card-name embeddings, card-image embeddings, and third-party meta/usage info. This lets the model represent and rank cards it never saw in training.

**How it applies:** This is the most directly relevant prior art for generalization to unseen cards. Reports ~55% accuracy predicting human picks on completely unseen cards (an earlier contrastive variant cited at ~45% zero-shot). Choice of representation barely affects known-card accuracy but greatly improves unseen-card performance - empirical evidence that feature/text-based card embeddings are the lever for generalization.

**Tradeoffs:** Unseen-card accuracy still well below in-distribution; requires curating multi-modal card features (text/image/meta) and a pipeline to fetch them for new cards; meta features can leak future information if not handled carefully.

### UrzaGPT - LoRA-tuned LLMs for card selection  
*Maturity: experimental*

arXiv:2508.08382 (2025). Fine-tunes open-weight LLMs (Mistral-7B-Instruct, Llama-3-8B) with LoRA (rank 8, alpha 16) on ~1M picks from the 17lands NEO dataset. Draft state is serialized as natural-language prompts ('My pool so far: [cards]; Current pack: [cards]; Which card should I pick?'). Cards represented by NAME ONLY (full text hurt performance). Best fine-tuned accuracy 66.2% (Llama-3-8B, ~10k steps); Mistral 64.3%; GPT-4o zero-shot 43%; cites a domain-specific SOTA at ~68%.

**How it applies:** Tests whether pretrained LLM knowledge of card names/semantics enables drafting and easier adaptation to new expansions. Because cards are referenced by name in text, the LLM's pretraining can in principle transfer to cards/sets outside the fine-tuning data, but the authors explicitly restrict evaluation to NEO and flag transfer/generalization as unresolved.

**Tradeoffs:** Expensive (3xA100, 6h/model), slower inference than a small MLP, and generalization to unseen sets is asserted-plausible but not measured. Name-only representation discards mechanical text.

### CubeCobra production draftbot (encoder-decoder embedding net)  
*Maturity: production-proven*

Documented in CubeCobra's official DraftBot Primer articles. Per CubeCobra's own descriptions, the bot encodes the current pool into a 128-dim embedding (archetype/strategy), concatenates a 32-dim cube-context vector, and feeds the 160-dim input to a 'draft decoder' that scores every card in the pack; trained on hundreds of thousands of real human picks. Earlier documented design used a set of hand-defined 'oracles' (subproblems) combined by learned, time-varying weights. Falls back to simple card-presence vectors if the model is unavailable.

**How it applies:** Real-world deployed CCG draftbot that must handle arbitrary, user-built cubes including custom/unofficial cards - so generalization to out-of-distribution cards is a first-class production requirement, addressed via card embeddings plus a cube-context summary rather than per-cube training.

**Tradeoffs:** Architecture details come from CubeCobra's own docs/articles (not a peer-reviewed paper), so specifics are less verifiable; tuned for cube/synergy drafting which differs from set/limited drafting.

### Statistical Drafting (17lands-trained MLP with pack-mask)  
*Maturity: production-proven*

statisticaldrafting.com 'About' page. A small PyTorch MLP that takes a collection vector and a pack vector and outputs a per-card rating vector; the pack vector acts as a mask zeroing ratings for cards not in the pack (ones-vector at inference to rate the whole set). Adds BatchNorm, dropout, exponential LR decay; ~20 min to train 100-150 epochs on CPU. Trained on 17lands public premier-draft data filtered to high-volume, high-win-rate players. ~70% agreement with top-player picks for a typical set.

**How it applies:** Practical, lightweight set-specific pick-order tool trained on real high-skill 17lands data (vs Draftsim's casual data). Demonstrates the value of filtering to strong players as the training signal.

**Tradeoffs:** Per-set retraining (new model ~2 weeks after set launch); one-hot-style card vectors, so no explicit cross-set/unseen-card generalization. Source is a project page, not peer-reviewed.

### madrury/mtg-draftbot (interpretable archetype-weight model)  
*Maturity: experimental*

GitHub repo (madrury/mtg-draftbot). Linear/bilinear model: cards carry archetype weight vectors over color-pair archetypes; drafter archetype preference = held-cards @ card_archetype_weights; card preference = dot(card_archetype_weights, drafter_preferences); softmax to pick probabilities; trained with NLLLoss. No hidden layers, no attention, no learned per-card embeddings beyond the archetype weight matrix. Demonstrated mainly on simulated drafts.

**How it applies:** An interpretable, mechanistic take on synergy-aware drafting that explicitly models archetype commitment. Useful as a transparent baseline; cannot extrapolate to novel cards without precomputed archetype weights.

**Tradeoffs:** Validated largely on self-generated simulated data (circular), no human-data results reported, no generalization to unseen cards.

### khakhalin/MTG (official Draftsim paper code)  
*Maturity: research-validated*

GitHub repo accompanying the Draftsim paper (Ward et al.). Implements RandomBot, RaredraftBot, DraftsimBot, BayesBot, NNetBot; data pipeline to standardize Draftsim drafts and build CV splits; MDS scaling on co-drafting distances for exploratory card embeddings; bot_tester.py for evaluation.

**How it applies:** Reference implementation and reproducibility artifact for the canonical dataset and baselines; the co-drafting-distance/MDS analysis is an early form of unsupervised card embedding from co-pick statistics.

**Tradeoffs:** Research code; embeddings here are analytical (MDS) rather than the predictive driver; tied to the fixed Draftsim card set.

## Recommendations

- For your stated goal (generalization to UNSEEN cards), the directly-on-point prior art is Bertram et al. 2024 'Learning With Generalised Card Representations' (arXiv:2407.05879) plus the CPR base (2105.11864). The actionable lesson: do NOT use one-hot card IDs; represent each card by features the model can compute for any new card - card text embeddings, numeric/nominal attributes, name/image embeddings - and learn in an embedding/preference space. This is what moves you from 0% to ~55% on unseen cards.
- Use the Draftsim dataset (arxiv 2009.00655 / khakhalin/MTG / IEEE DataPort) and 17lands public data as your two main training corpora. Draftsim is large but casual/simulated; 17lands lets you filter to high-skill players (as Statistical Drafting does). Filtering to strong players is a cheap, proven way to improve the training signal.
- Adopt the standard task framing and metrics so you are comparable: predict the human pick from (pool, pack); report top-1 pick accuracy AND mean pick distance (rank gap between chosen and predicted). Known anchor numbers: Draftsim NN ~48.7% / 1.48 distance, CPR >56% relative improvement, generalized-representation ~55% on unseen cards, UrzaGPT 66.2% in-set.
- Decide architecture by generalization need: (a) one-hot MLP or masked-rating MLP (Statistical Drafting style) - simplest, strongest in-set, zero cross-set transfer; (b) Siamese/contrastive embedding net (CPR/InfoNCE) - best accuracy + synergy + extensible to features; (c) feature/text-based embedding net - the only one with demonstrated unseen-card transfer; (d) LoRA-tuned LLM (UrzaGPT) - leverages pretrained card-name knowledge and adapts fast to new sets, but transfer is unproven and serving is costlier.
- If considering the LLM route, note UrzaGPT found NAME-ONLY card prompts beat full-text prompts and reached 66.2% with Llama-3-8B; GPT-4o zero-shot was only 43%. A small fine-tuned model is both cheaper and better than zero-shot frontier models here.
- Steal CubeCobra's production trick for out-of-distribution pools: summarize the overall card environment (their 32-dim 'cube-context' vector) and concatenate it with the pool embedding, so the same model adapts across very different card pools without retraining per environment.

## Open questions

- No published work convincingly measures cross-SET transfer for the LLM approach: UrzaGPT restricts to the NEO set and only asserts that performance 'should translate.' Whether name-only LLM drafting truly generalizes to a brand-new expansion is unverified.
- Generalized card representations top out around ~55% on unseen cards vs ~66-70% in-distribution - the unseen-card accuracy gap is still large and there is no consensus on which representation (text vs numeric vs image vs meta) or which fusion drives the gains.
- Imitation accuracy (matching human picks) is not the same as draft/deck WIN-RATE quality; the field largely optimizes pick-matching, and the relationship between higher pick-accuracy and better resulting decks is under-studied (Witten's blog even found a larger but lower-skill dataset gave higher accuracy yet worse picks).
- Meta/usage features (e.g., 17lands win-rate stats) boost unseen-card prediction but risk leaking future/aggregate information; clean evaluation protocols that prevent leakage for genuinely 'new' cards are not standardized.
- I could not text-extract several arXiv PDFs (2009.00655, 2105.11864, 2407.05898) directly; their architecture details here come from the HTML/Medium reimplementation, author abstracts, and Draftsim's own write-up. The CPR exact accuracy figures and the InfoNCE paper's numbers should be confirmed against the original PDFs before being quoted precisely.
- No large public Kaggle COMPETITION specifically for MtG draft-pick prediction was found - only general MtG datasets (e.g., mtgtop8 decklists) and the academic Draftsim/17lands datasets. If a Kaggle competition is needed as prior art, it appears not to exist as of this search.

## Sources

- [AI solutions for drafting in Magic: the Gathering (Ward et al., IEEE CoG 2021)](https://arxiv.org/abs/2009.00655) *(paper)* — Released ~107,949 Draftsim human drafts; NNetBot (512-neuron dense layers + dropout, softmax over N one-hot cards, cross-entropy) won at 48.67% mean per-pick accuracy / 1.48 pick distance vs DraftsimBot 44.54% and BayesBot 43.36%. One-hot cards => no unseen-card generalization (noted as future work).
- [Teaching an AI to Draft Magic: the Gathering (Zachary Witten, TDS/Medium)](https://medium.com/data-science/teaching-an-ai-to-draft-magic-the-gathering-ba38b6a3d1f3) *(blog)* — Concrete Draftsim-style architecture: input = 2N vector (pack one-hot + collection counts), 512-neuron dense layers with dropout, N-way softmax, one-hot human-pick target; 59% accuracy vs 50% linear baseline. Discusses (but does not implement) MTGJSON features for new-set generalization.
- [Predicting Human Card Selection in MtG with Contextual Preference Ranking (Bertram, Fürnkranz, Müller, IEEE CoG 2021)](https://arxiv.org/abs/2105.11864) *(paper)* — Siamese network trained on (context, chosen, rejected) triplets; CPR learns an embedding where chosen cards sit closer to deck context. >56% relative accuracy gain and >83% lower pick distance vs Draftsim NN. Original version still one-hot (fixed card set).
- [Learning With Generalised Card Representations for Magic: The Gathering (Bertram, Fürnkranz, Müller, IEEE CoG 2024)](https://arxiv.org/abs/2407.05879) *(paper)* — Most relevant to unseen-card generalization: replaces one-hot IDs with numeric/nominal features, card-text, name, image, and meta embeddings; ~55% accuracy on completely unseen cards. Representation choice barely affects known cards but greatly improves new cards. Best-paper nominee.
- [UrzaGPT: LoRA-Tuned LLMs for Card Selection in CCGs (2025)](https://arxiv.org/abs/2508.08382) *(paper)* — LoRA fine-tunes (Llama-3-8B, Mistral-7B) on ~1M 17lands NEO picks; cards by name only in NL prompts; best 66.2% (Llama-3-8B), Mistral 64.3%, GPT-4o zero-shot 43%, cited SOTA ~68%. Cross-set generalization asserted plausible but explicitly untested (restricted to NEO).
- [Statistical Drafting - About](https://statisticaldrafting.com/about) *(docs)* — PyTorch MLP: collection + pack vectors -> per-card ratings, pack vector as mask; BatchNorm/dropout/exp-decay LR; trained on 17lands high-skill players; ~70% agreement with top-player picks; per-set retraining, no cross-set generalization.
- [khakhalin/MTG - Draftsim paper bots and embeddings (GitHub)](https://github.com/khakhalin/MTG) *(repo)* — Official Draftsim code: RandomBot/RaredraftBot/DraftsimBot/BayesBot/NNetBot, Draftsim data standardization + CV pipeline, MDS scaling on co-drafting distances as exploratory card embeddings.
- [madrury/mtg-draftbot - Algorithmic Drafting for MtG (GitHub)](https://github.com/madrury/mtg-draftbot) *(repo)* — Interpretable bilinear archetype-weight model (held-cards @ archetype_weights, softmax, NLLLoss); no hidden layers/attention/learned embeddings; demonstrated on simulated drafts; no unseen-card generalization.
- [Official CubeCobra DraftBot Primer (Parts 1-2)](https://cubecobra.com/content/article/6008b17565264010480a4a84) *(blog)* — Per CubeCobra docs, draftbot encodes pool to a 128-dim embedding + 32-dim cube-context => 160-dim decoder input scoring each pack card; trained on hundreds of thousands of human picks; earlier 'oracles + learned time-varying weights' design; must handle arbitrary/custom cube cards.
- [Contrastive Learning of Preferences with a Contextual InfoNCE Loss (arXiv:2407.05898)](https://arxiv.org/abs/2407.05898) *(paper)* — Companion/methodological work extending CPR to a contextual InfoNCE (multi-candidate contrastive) loss for preference ranking; relevant to scaling Siamese preference learning beyond single triplets. PDF body not text-extractable; details taken from search-surfaced abstract, treat numbers cautiously.

## Source verification

**Overall confidence:** high. All 10 cited URLs are real and reachable, and every source is accurately attributed at the level this fact-check could verify. No hallucinated or fabricated sources were found. The five arXiv papers all exist with the stated titles/authors (notably Bertram/Fürnkranz/Müller author four of the academic works). The two GitHub repos and the Statistical Drafting page were confirmed in detail and match their descriptions verbatim. The main residual risk is in precise QUANTITATIVE claims drawn from arXiv PDF bodies that are not present in the abstracts: the CPR >56%/>83% gains, the Draftsim per-bot accuracies and exact 107,949 draft count, and several UrzaGPT internals (dataset size, Mistral 64.3%, name-only finding, NEO restriction). These are all plausible and internally consistent, and the researcher transparently flags the unverifiable ones in openQuestions, but they were not independently confirmed here and should be checked against the full PDFs before being quoted as exact. The CubeCobra architecture dimensions are only weakly corroborated (search snippet mirroring the claim; article body not extractable). The qualitative narrative of the report — supervised imitation framing, one-hot vs feature/text embeddings as the generalization axis, CPR/Siamese as the big accuracy jump, 2407.05879 as the on-point unseen-card work (~55%), and the open question that pick-accuracy != win-rate — is well supported by the confirmed abstracts. No skeptical red flags beyond the unverified exact numbers noted above."

**Flagged claims:**
- ⚠️ CPR (2105.11864) headline numbers '>56% relative accuracy improvement' and '>83% reduction in pick distance' could NOT be confirmed from the arXiv abstract (only the qualitative method is in the abstract). They come from the full PDF. The researcher does flag this in open questions, but these load-bearing figures remain unverified by this fact-check.
- ⚠️ Draftsim (2009.00655) exact figures '48.67% / 1.48 pick distance', 'DraftsimBot 44.54%', 'BayesBot 43.36%' are not in the abstract and were not independently verified; only the qualitative ranking (NN wins; Bayes/expert beat simple heuristics) is confirmed. The precise dataset count '107,949' is also not in the abstract (paper says 'over 100,000') — researcher already notes this discrepancy.
- ⚠️ UrzaGPT (2508.08382) specifics not in the abstract and unverified at the numeric level: '~1M NEO picks' dataset size, the Mistral-7B '64.3%' figure, the name-only-beats-full-text finding, LoRA rank 8/alpha 16, and the 'restricted to NEO / cross-set untested' claim. Only 66.2%, GPT-4o 43%, and 'not reaching domain-specific SOTA' are confirmed from the abstract.
- ⚠️ CubeCobra exact dimensions (128-dim pool / 32-dim cube-context / 160-dim decoder) rest on a web-search snippet whose wording closely mirrors the researcher's own; the source article body was not directly extractable. Treat the precise dimensionality as lightly corroborated, not firmly verified.

**Checked sources:**

| URL | Reachable | Accurate | Note |
|---|---|---|---|
| https://arxiv.org/abs/2009.00655 | ✅ | ✅ | Confirmed real: 'AI solutions for drafting in Magic: the Gathering' by Ward, Brooks, Troha, Mills, Khakhalin. Abstract confirms 'over 100,000 simulated, anonymized human drafts from Draftsim.com', four/five-agent benchmark (heuristic, expert-heuristic, Naive Bayes, deep NN), and that the NN agent wins. The specific architecture details (512-neuron layers, dropout) and exact numbers (48.67%/1.48 distance, DraftsimBot 44.54%, BayesBot 43.36%) are NOT in the abstract and could not be independently verified here; they come from the full PDF and are consistent with the companion code/Medium reimplementation. The precise '107,949' figure also is not stated in the abstract (paper says 'over 100,000'), which the researcher correctly flags. |
| https://medium.com/data-science/teaching-an-ai-to-draft-magic-the-gathering-ba38b6a3d1f3 | ✅ | ✅ | Confirmed: Zachary Witten. Two 512-neuron dense layers with dropout, 2N input (pack one-hot + collection counts), N-way softmax, ~59% accuracy vs ~50% linear baseline, 250 cards (GRN). Discusses MTGJSON for new-set generalization without implementing it. Also confirms the dataset-quality nuance (larger lower-skill Draftsim data gave worse picks than Flooey data) cited in the open questions. |
| https://arxiv.org/abs/2105.11864 | ✅ | ✅ | Confirmed real: 'Predicting Human Card Selection in Magic: The Gathering with Contextual Preference Ranking' by Bertram, Fürnkranz, Müller. Abstract confirms the contextual preference network that compares two deck extensions. However, the specific quantitative claims (>56% relative accuracy improvement, >83% pick-distance reduction) and the Siamese/triplet-loss + one-hot details are NOT visible in the abstract and could not be independently confirmed; they originate from the full PDF. Researcher already flags these numbers as needing PDF confirmation. |
| https://arxiv.org/abs/2407.05879 | ✅ | ✅ | Fully confirmed from abstract: 'Learning With Generalised Card Representations for Magic: The Gathering' by Bertram, Fürnkranz, Müller. Numerical/nominal/text features, card images, meta usage info; 'predict 55% of human choices on completely unseen cards'; representation choice has little effect on known cards but greatly improves unseen cards; IEEE CoG 2024 best-paper nominee. All accurately described. |
| https://arxiv.org/abs/2508.08382 | ✅ | ✅ | Confirmed real: 'UrzaGPT: LoRA-Tuned Large Language Models for Card Selection in Collectible Card Games' by Timo Bertram. Abstract confirms LoRA fine-tuning, 66.2% accuracy at 10,000 steps, GPT-4o zero-shot 43%, and 'not reaching the capability of domain-specific models' (the cited ~68% SOTA). The dataset size (~1M NEO picks), specific models (Llama-3-8B vs Mistral-7B 64.3%), name-only representation, and NEO restriction are NOT in the abstract; they come from the full paper. Plausible and consistent but not independently verified at the numeric level. |
| https://statisticaldrafting.com/about | ✅ | ✅ | Fully confirmed verbatim: PyTorch MLP, collection+pack vectors -> card rating vector, BatchNorm/dropout/exp-decay LR, ~20 min for 100-150 epochs on CPU, trained on high-volume/high-win-rate 17lands players, ~70% agreement with top-player picks, models released no earlier than 2 weeks after set release. Accurate. |
| https://github.com/khakhalin/MTG | ✅ | ✅ | Confirmed: official Draftsim paper code. Implements RandomBot, RaredraftBot, DraftsimBot, BayesBot, NNetBot; includes standardization/CV notebooks, MDS scaling on co-drafting distances, and bot_tester.py. Accurate. |
| https://github.com/madrury/mtg-draftbot | ✅ | ✅ | Confirmed: interpretable bilinear archetype-weight model (drafter preference = held cards @ archetype weights; card preference = dot product; softmax; NLLLoss); no hidden layers/attention/learned embeddings beyond archetype weights; demonstrated on simulated drafts; no human-data results; no unseen-card generalization. Accurate, including the 'circular/simulated-only' caveat. |
| https://cubecobra.com/content/article/6008b17565264010480a4a84 | ✅ | ✅ | Page reachable (title 'Official CubeCobra DraftBot Primer: Part 2 The Basics') but article body was NOT text-extractable via WebFetch. A web search corroborated the architecture claims: 128-dim pool embedding + 32-dim cube-context vector = 160-dim decoder input scoring pack cards, trained on hundreds of thousands of human picks, and an oracle+time-varying-weights subproblem design. Caveat: the search-result phrasing closely mirrors the researcher's own wording, so this is weak independent corroboration; the researcher correctly notes these come from CubeCobra's own non-peer-reviewed docs. |
| https://arxiv.org/abs/2407.05898 | ✅ | ✅ | Confirmed real: 'Contrastive Learning of Preferences with a Contextual InfoNCE Loss' by Bertram, Fürnkranz, Müller. Abstract confirms it adapts CLIP's InfoNCE loss for contextual preference learning over card pools (handles multiple positives per batch, avoids triplet mining). Accurately described as a methodological extension of CPR. Researcher appropriately flags that specific numbers should be treated cautiously. |
