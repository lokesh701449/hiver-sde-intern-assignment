# Engineering and Architecture Decision Log

This log documents key technical, architectural, and evaluation design decisions made during the development of the AmazonHelp AI Support Agent system.

---

### Decision 1: Conversation-Level Context vs. Individual Tweet Evaluation
- **Decision**: Evaluate system performance on complete multi-turn conversation threads rather than isolated customer tweets.
- **Why**: Isolated customer tweets often lack crucial context (e.g. "@AmazonHelp where is it?" vs prior turn stating "Order #12345 sent yesterday"). Support agents and automation require full context to determine intent and escalation status accurately.
- **Tradeoff**: Increases payload size and prompt token length, requiring efficient context truncation and retrieval strategies.

### Decision 2: Domain Restriction to `@AmazonHelp` Traffic
- **Decision**: Scope the dataset and historical retrieval exclusively to `@AmazonHelp` customer support interactions within TWCS.
- **Why**: Twitter Customer Support (TWCS) contains multi-brand data with diverse domain conventions. AmazonHelp is the largest contiguous e-commerce brand dataset (~374k tweets, ~82k threads), providing rich, domain-consistent e-commerce patterns.
- **Tradeoff**: Model heuristics and retrieval index are tailored to Amazon e-commerce workflows; adapting to telecom or flight support requires re-indexing and domain tuning.

### Decision 3: Custom 11-Intent E-Commerce Support Taxonomy
- **Decision**: Define a custom 11-class intent taxonomy (`delivery_shipping_delay`, `returns_refund_inquiry`, `item_condition_issue`, `device_hardware_support`, `prime_membership_billing`, `digital_streaming_services`, `account_access_security`, `payment_promo_giftcard`, `order_modification_cancellation`, `marketplace_third_party_seller`, `unclear_other`).
- **Why**: Public TWCS data lacks ground-truth intent labels. A structured 11-intent taxonomy captures actionable e-commerce customer support workflows without over-granularity.
- **Tradeoff**: Class imbalance is inherent in customer support traffic (e.g., delivery inquiries outnumber cancellation requests 10:1).

### Decision 4: Inclusion of `unclear_other` Fallback Intent
- **Decision**: Explicitly retain `unclear_other` in the taxonomy rather than forcing every ambiguous tweet into a specific functional intent.
- **Why**: Real customer messages are frequently incomplete, vague ("help me"), or noise. Forcing ambiguous queries into functional intents distorts automation signals.
- **Tradeoff**: Classifying vague queries as `unclear_other` can lower overall recall on minor intents if over-predicted.

### Decision 5: Stratified 80/10/10 Data Split Strategy
- **Decision**: Use an 80/10/10 conversation-level split for index building, development/golden set, and frozen heldout evaluation.
- **Why**: Ensures zero data leakage across historical retrieval evidence, development prompt engineering, and final benchmark evaluation.
- **Tradeoff**: Heldout evaluation is constrained to 200 heldout conversations to allow 100% human-verified golden labelling.

### Decision 6: 100% Human-Reviewed 200-Example Golden Set
- **Decision**: Manually review and audit all 200 development set pre-labels.
- **Why**: AI-generated or heuristic pre-labels contain systematic bias. Human auditing confirmed 93 labels and corrected 107, establishing a reliable ground truth.
- **Tradeoff**: High initial manual effort required before development and evaluation could begin.

### Decision 7: Frozen 200-Example Heldout Test Set with Seed Locking
- **Decision**: Create a dedicated 200-example heldout set (`heldout_test_final.csv`, seed 2026) that remained strictly frozen and untouched during system development.
- **Why**: Prevents overfitting to dev examples and ensures unbiased, leak-free heldout benchmark evaluation.
- **Tradeoff**: System design decisions were made strictly on dev set feedback without iterating on heldout metrics.

### Decision 8: Strict Conversation-ID Exclusion During Historical Retrieval
- **Decision**: Exclude the current evaluation conversation ID from FAISS vector retrieval.
- **Why**: In evaluation pipelines, retrieving an example's own historical turns causes severe retrieval leakage and artificially inflates groundedness/accuracy metrics.
- **Tradeoff**: Requires explicit ID filtering logic in `retrieval.py` during evaluation runs.

### Decision 9: Historical Retrieval for Resolution/Style Grounding, Not Label Ground-Truth
- **Decision**: Use retrieved historical AmazonHelp evidence to guide response generation and tone, but NOT to override classifier/policy decisions.
- **Why**: Historical tweets reflect past public agent responses, which may be outdated or incomplete. The current conversation context must drive intent and escalation.
- **Tradeoff**: Avoids copy-pasting historical answers that may not apply to the current customer's specific issue.

### Decision 10: Decoupling LLM Generation from Deterministic Escalation Policy (Hybrid RAG)
- **Decision**: Rely on the local LLM (`qwen2.5:3b`) for intent classification and public reply drafting, but use a deterministic policy for final escalation decisions.
- **Why**: Pure LLM escalation predictions exhibited heavy bias (e.g. predicting `escalate=False` for all cases despite explicit prompts). Deterministic policy rules provide predictable escalation behavior and improve safety for sensitive customer actions, while heldout evaluation shows 95.0% escalation recall.
- **Tradeoff**: Requires rule definition based on domain signals (order lookup, DM requests, damage reports).

### Decision 11: Prioritizing Intent Macro F1 Alongside Accuracy
- **Decision**: Evaluate intent performance using Macro F1 alongside overall Accuracy.
- **Why**: In imbalanced datasets, accuracy can be artificially inflated by over-predicting the majority class (`delivery_shipping_delay`). Macro F1 enforces equal weight across all 11 intent classes.
- **Tradeoff**: Highlights low performance on rare classes (e.g., `order_modification_cancellation`), accurately reflecting real-world tail risks.

### Decision 12: Treating Transformer Results as DEV/CV Evidence Only
- **Decision**: Exclude fine-tuned DistilBERT metrics from the primary heldout benchmark table because DistilBERT was evaluated on an 80/20 dev split (n=40 val) rather than the frozen heldout benchmark.
- **Why**: Mixing cross-validation or dev validation metrics with strict heldout benchmark results violates evaluation rigor.
- **Tradeoff**: Prevents misrepresenting dev performance as heldout performance.

### Decision 13: Treating LLM-as-Judge as Secondary Evaluator
- **Decision**: Use LLM-as-judge as a complementary evaluation tool while relying on human agreement studies to validate alignment.
- **Why**: The human agreement study revealed low kappa on subjective dimensions (Relevance: 0.0598, Tone: 0.0670), demonstrating that LLM judges cannot replace human evaluation.
- **Tradeoff**: Requires transparently reporting agreement limits rather than claiming LLM judge scores are authoritative.
