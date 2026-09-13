# Sampling Strategy Note — 50 Representative Heldout Examples

## Sampling Protocol
The 50 examples were selected deterministically from `heldout_test_final.csv` (n=200) using stratified sampling to ensure comprehensive coverage across:

1. **Intent Taxonomy**: Represents all 11 intent classes.
2. **Difficulty Stratification**: Covers Easy, Medium, and Hard cases.
3. **Escalation Decisions**: Includes both `Escalate=True` and `Escalate=False` examples.
4. **Classification Accuracy**: Includes both correctly classified and misclassified examples.

## Sample Composition

| Attribute | Distribution in 50 Sample Examples |
| :--- | :---: |
| Total Selected Examples | 50 |
| Escalated Cases (`True`) | 39 |
| Non-Escalated Cases (`False`) | 11 |
| Difficulty: `Easy` | 25 |
| Difficulty: `Hard` | 5 |
| Difficulty: `Medium` | 20 |

### Intent Breakdown in Sample:
- `delivery_shipping_delay`: 9 examples
- `returns_refund_inquiry`: 5 examples
- `payment_promo_giftcard`: 5 examples
- `device_hardware_support`: 5 examples
- `digital_streaming_services`: 5 examples
- `prime_membership_billing`: 4 examples
- `order_modification_cancellation`: 4 examples
- `item_condition_issue`: 4 examples
- `account_access_security`: 4 examples
- `marketplace_third_party_seller`: 3 examples
- `unclear_other`: 2 examples
