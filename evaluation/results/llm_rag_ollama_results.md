# LLM + RAG Evaluation — Development Set Results (V1)

> [!IMPORTANT]
> **Scope**: Development set only (`golden_set_final.csv`, n=200). `heldout_test_final.csv` was **NOT accessed**.
> **Prompt version**: V1 (fixed; no tuning based on individual development errors).
> **Provider**: `ollama/qwen2.5:3b`

## 1. All-Approaches Comparison (Development Estimates)

> [!WARNING]
> Metrics from different protocols (full dev-set, CV, 80/20 split) are NOT directly comparable.
> Each row is labelled with its evaluation protocol.

| Approach | Intent Acc | Macro F1 | Esc Acc | Esc F1 | Combined | Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| Rule-Based V2 (dev full-set) | 61.50% | 0.5495 | 78.50% | 0.8436 | 52.00% | Full 200-example dev set evaluation |
| TF-IDF + LR (5-fold CV dev) | 57.50% | 0.4279 | 80.50% | — | — | 5-fold stratified CV on 200 dev examples |
| DistilBERT (5-fold CV dev) | 42.00% | 0.2505 | 67.50% | — | 31.50% | 5-fold CV on 200 dev examples (high variance ±5%) |
| DistilBERT (dev 80/20 split) | 75.00% | 0.6959 | 65.00% | 0.7813 | 47.50% | Single 80/20 stratified split (n=40 val; may be high-variance) |
| **LLM+RAG / Gemini 3.6 Flash (V1)** | **66.00%** | **0.6046** | **32.50%** | **0.0000** | **17.00%** | Full 200-example dev set |

## 2. Per-Intent Metrics (LLM+RAG, Dev Set)

| Intent | Support | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: | :---: |
| `account_access_security` | 17 | 0.9375 | 0.8824 | **0.9091** |
| `device_hardware_support` | 8 | 0.8750 | 0.8750 | **0.8750** |
| `digital_streaming_services` | 16 | 0.7647 | 0.8125 | **0.7879** |
| `returns_refund_inquiry` | 21 | 0.7391 | 0.8095 | **0.7727** |
| `payment_promo_giftcard` | 13 | 0.8889 | 0.6154 | **0.7273** |
| `delivery_shipping_delay` | 56 | 0.5283 | 1.0000 | **0.6914** |
| `prime_membership_billing` | 8 | 0.8000 | 0.5000 | **0.6154** |
| `marketplace_third_party_seller` | 4 | 1.0000 | 0.2500 | **0.4000** |
| `item_condition_issue` | 24 | 0.6667 | 0.2500 | **0.3636** |
| `unclear_other` | 27 | 1.0000 | 0.1481 | **0.2581** |
| `order_modification_cancellation` | 6 | 0.5000 | 0.1667 | **0.2500** |

## 3. Top Intent Confusion Pairs (LLM+RAG)

| Rank | Actual | Predicted | Count |
| :---: | :--- | :--- | :---: |
| 1 | `unclear_other` | `delivery_shipping_delay` | 17 |
| 2 | `item_condition_issue` | `delivery_shipping_delay` | 15 |
| 3 | `prime_membership_billing` | `delivery_shipping_delay` | 4 |
| 4 | `returns_refund_inquiry` | `delivery_shipping_delay` | 3 |
| 5 | `order_modification_cancellation` | `delivery_shipping_delay` | 3 |
| 6 | `unclear_other` | `digital_streaming_services` | 3 |
| 7 | `digital_streaming_services` | `delivery_shipping_delay` | 2 |
| 8 | `payment_promo_giftcard` | `delivery_shipping_delay` | 2 |
| 9 | `payment_promo_giftcard` | `returns_refund_inquiry` | 2 |
| 10 | `item_condition_issue` | `returns_refund_inquiry` | 2 |

## 4. Escalation Confusion Matrix

| | Predicted Escalate | Predicted No Escalate |
| :--- | :---: | :---: |
| **Actual Escalate** | 0 (TP) | 135 (FN) |
| **Actual No Escalate** | 0 (FP) | 65 (TN) |

## 5. Runtime & Retrieval Stats

| Metric | Value |
| :--- | :---: |
| Total runtime | 916s |
| API calls made | 200 / 200 |
| Failed LLM calls | 0 |
| Avg LLM latency | 4080 ms |
| Avg retrieval latency | 374.6 ms |
| Avg retrieved similarity | 0.6995 |
| Avg LLM confidence | 0.919 |
| Confidence (min / max) | 0.750 / 0.980 |
| Confidence (median) | 0.950 |

## 6. Qualitative Analysis — Retrieval Usefulness

### 6a. Examples Where Retrieved Cases Appear USEFUL

**Example 1** (id=1, similarity=0.8031)

- Customer: `@AmazonHelp Order # 407-7008362-4966709 Amazon Tracking  says Attempted delivery at 11.38 PM - (cont) https://t.co/4iN0X...`
- Human intent: `delivery_shipping_delay` → Predicted: `delivery_shipping_delay` ✅
- Top retrieved case: `Order # 407-7008362-4966709 Amazon Tracking  says Attempted delivery at 11.38 PM - (cont) Have submitted the details on ...`
- AmazonHelp replied: `Could you please let us know what went wrong? We'll definitely look into it. 2/2 ^PS...`

**Example 2** (id=3, similarity=0.7818)

- Customer: `@AmazonHelp Tracking ID is Q80674748604...`
- Human intent: `delivery_shipping_delay` → Predicted: `delivery_shipping_delay` ✅
- Top retrieved case: `@AmazonHelp @115830 my item was meant to be delivered today, I've waited all day,where is it? Tracking #Q40721330864...`
- AmazonHelp replied: `I'm sorry your parcel hasn't been delivered yet. Have you received any delay notifications or an upd...`

**Example 3** (id=4, similarity=0.7749)

- Customer: `@AmazonHelp Thanks. I've already checked this - the carrier was Amazon Shipping, it's saying to contact customer service...`
- Human intent: `delivery_shipping_delay` → Predicted: `delivery_shipping_delay` ✅
- Top retrieved case: `hi. My parcel says it was delivered and nothing has arrived. What do I do? Thanks. Not sure which carrier to contact as ...`
- AmazonHelp replied: `I'm sorry to hear this, but we're here for you! Try these steps and let us know if this doesn't help...`

**Example 4** (id=6, similarity=0.7534)

- Customer: `@AmazonHelp Lasership. And I don't have my package...`
- Human intent: `delivery_shipping_delay` → Predicted: `delivery_shipping_delay` ✅
- Top retrieved case: `@AmazonHelp I do not my package will not get here in time of LaserShip right before 9:00 PM come...`
- AmazonHelp replied: `Does the tracking show 'out for delivery'?  ^RO...`

**Example 5** (id=8, similarity=0.7716)

- Customer: `@AmazonHelp The delivery man is not good 
He is stupid pls hire such kind of people .
They are just degrading your promi...`
- Human intent: `delivery_shipping_delay` → Predicted: `delivery_shipping_delay` ✅
- Top retrieved case: `@115830 @AmazonHelp Made order next Saturday failed to be delivered on Wednesday and Thursday rang yesterday and this af...`
- AmazonHelp replied: `Hi Raj, without going into any personal account info could you please tell us more about this? ^HS...`

### 6b. Examples Where Retrieval Appears IRRELEVANT or MISLEADING

**Example 1** (id=2, similarity=0.7020)

- Customer: `@AmazonHelp I went through everything just to be told you can’t help. The order is late and the codes don’t work. I thou...`
- Human intent: `unclear_other` → Predicted: `delivery_shipping_delay` ❌
- Top retrieved case: `This is 3rd time in same month, amazon did blunder with pathetic service with me. I will be running compaign against @Am...`
- Analysis: Retrieved case may have biased the LLM toward `delivery_shipping_delay`

**Example 2** (id=5, similarity=0.7919)

- Customer: `@AmazonHelp Customer service has already helped as much as they can. Stop using me as a guinea pig in trying to get the ...`
- Human intent: `unclear_other` → Predicted: `delivery_shipping_delay` ❌
- Top retrieved case: `Hi .it's frustrating trying to contact .@AmazonHelp  customer service for delivery related issues....`
- Analysis: Retrieved case may have biased the LLM toward `delivery_shipping_delay`

**Example 3** (id=10, similarity=0.7273)

- Customer: `@AmazonHelp The team picked up the return package. Thank you!...`
- Human intent: `returns_refund_inquiry` → Predicted: `delivery_shipping_delay` ❌
- Top retrieved case: `I placed a return of order#405-5827927-7849934 on 5th november, 2017. But it hasn't been picked up yet and no replies ca...`
- Analysis: Retrieved case may have biased the LLM toward `delivery_shipping_delay`

**Example 4** (id=11, similarity=0.6923)

- Customer: `@AmazonHelp No, there's no point now. I had to wait an entire week for a "next day delivery", so I'm not going to wait a...`
- Human intent: `item_condition_issue` → Predicted: `delivery_shipping_delay` ❌
- Top retrieved case: `how is it that with Amazon prime I won't get deliveries until next Wednesday??...`
- Analysis: Retrieved case may have biased the LLM toward `delivery_shipping_delay`

**Example 5** (id=13, similarity=0.7512)

- Customer: `@AmazonHelp I'm not certain, so I don't want to blame anyone! But one of the issues definitely wasn't down to the carrie...`
- Human intent: `item_condition_issue` → Predicted: `delivery_shipping_delay` ❌
- Top retrieved case: `All I want is estimated delivery times and to know why my items haven't been sent as one package. There is also an item ...`
- Analysis: Retrieved case may have biased the LLM toward `delivery_shipping_delay`


## 7. Summary Answers

| Question | Answer |
| :--- | :--- |
| A. Provider/model | `ollama/qwen2.5:3b` |
| B. Intent accuracy | **66.00%** |
| C. Macro F1 | **0.6046** |
| D. Escalation acc / F1 | **32.50% / 0.0000** |
| E. Combined accuracy | **17.00%** |
| F. Beats Rule V2? | ✅ YES (66.00% vs 61.50%) |
| G. Retrieval useful? | Marginal (avg similarity=0.6995) |
| H. Proceed to reply generation? | Yes — intent acc ≥ V2, retrieval grounding works |
