"""
escalation_policy.py
--------------------
Deterministic policy engine for customer support escalation decisions.

Evaluates whether a conversation requires private account/order lookup,
specialized investigation, or human intervention based on generic domain rules.
"""

import re
from typing import Dict, Any


def evaluate_escalation_policy(
    customer_message: str,
    conversation_context: str = "",
    predicted_intent: str = "unclear_other"
) -> Dict[str, Any]:
    """
    Evaluates the conversation context and predicted intent against a deterministic policy.

    Returns:
        {
            "escalate": bool,
            "escalation_reason": str
        }
    """
    text = (customer_message + " " + conversation_context).lower()

    # 1. Order Number / Specific Order Identifiers Check
    has_order_id = bool(
        re.search(r'\b(?:order|id|tracking|#)\s*#?\s*\d{3}-\d{7}-\d{7}\b', text, re.IGNORECASE) or
        re.search(r'\b\d{3}-\d{7}-\d{7}\b', text) or
        re.search(r'order\s*(?:#|number|no\.?)\s*:?\s*\d+', text, re.IGNORECASE)
    )

    # 2. Key phrases indicative of specific order/account investigation
    has_dm_request = any(k in text for k in [
        'dm us', 'direct message', 'send us a message', 'pm us', 'reach out via dm', 'private message'
    ])
    has_account_security = (predicted_intent == "account_access_security") or any(k in text for k in [
        'login', 'password', 'otp', '2fa', 'hacked', 'account locked', 'sign in', 'credential', 'locked out', 'compromised'
    ])
    has_refund_billing_issue = (predicted_intent in ["returns_refund_inquiry", "payment_promo_giftcard", "prime_membership_billing"]) or any(k in text for k in [
        'refund', 'double charge', 'charged twice', 'billing error', 'missing credit', 'cashback', 'gift card balance', 'unauthorized charge'
    ])
    has_marketplace = (predicted_intent == "marketplace_third_party_seller") or any(k in text for k in [
        'third party seller', 'third-party seller', 'a-to-z claim', 'marketplace seller'
    ])
    has_cancellation_mod = (predicted_intent == "order_modification_cancellation") or any(k in text for k in [
        'cancel order', 'change address', 'modify order', 'cancel my order'
    ])

    # 3. Delivery / Tracking specific logic
    is_delivery_intent = (predicted_intent == "delivery_shipping_delay")
    has_delivery_investigation_keywords = any(k in text for k in [
        'where is my order', 'where is my package', 'not arrived', 'attempted delivery',
        'marked delivered', 'showing delivered', 'fake delivery', 'delayed', 'late delivery',
        'tracking says', 'courier', 'carrier', 'tracking update', 'dispatch', 'expected delivery',
        'where is my', 'parcel', 'shipment'
    ])

    # Check if delivery is a generic policy question or general inquiry without specific order action
    is_generic_delivery_policy = any(k in text for k in [
        'delivery policy', 'how long does shipping take', 'shipping charges', 'free delivery rules', 'prime shipping terms'
    ])

    # 4. Damage / Condition specific logic
    is_damage_intent = (predicted_intent == "item_condition_issue")
    has_damage_resolution_request = any(k in text for k in [
        'refund', 'replace', 'replacement', 'compensation', 'damaged item', 'broken', 'wrong item', 'faulty', 'return'
    ])
    is_general_packaging_complaint = any(k in text for k in [
        'improve packaging', 'staff need training', 'feedback for fc', 'packaging quality is bad'
    ]) and not has_damage_resolution_request

    # 5. Device Support / Digital Services specific logic
    is_device_intent = (predicted_intent == "device_hardware_support")
    is_digital_intent = (predicted_intent == "digital_streaming_services")
    has_public_troubleshooting = any(k in text for k in [
        'reset', 'restart app', 'fastboot', 'wake word', 'black screen', 'crash', 'software update', 'how to setup'
    ])

    # ─── EVALUATION LOGIC ───────────────────────────────────────────────────────

    # Rule A: Order identifier or DM/private request present → Escalate
    if has_order_id:
        return {
            "escalate": True,
            "escalation_reason": "Specific order ID provided; requires order/account lookup in internal tools."
        }

    if has_dm_request:
        return {
            "escalate": True,
            "escalation_reason": "Requires DM / private channel communication for account details."
        }

    # Rule B: Delivery / Tracking investigation
    if is_delivery_intent or has_delivery_investigation_keywords:
        if is_generic_delivery_policy:
            return {
                "escalate": False,
                "escalation_reason": "General delivery policy inquiry; answerable with public information."
            }
        return {
            "escalate": True,
            "escalation_reason": "Delivery/tracking issue requiring specific order status lookup."
        }

    # Rule C: Account access / Security recovery
    if has_account_security:
        if any(k in text for k in ['sign out button', 'app navigation', 'how to logout']):
            return {
                "escalate": False,
                "escalation_reason": "General app navigation query; answerable with public guidance."
            }
        return {
            "escalate": True,
            "escalation_reason": "Account access or security issue requiring identity verification/investigation."
        }

    # Rule D: Refunds, Billing, Monetary disputes
    if has_refund_billing_issue:
        if any(k in text for k in ['refund policy', 'how do refunds work', 'what is return policy']) and not any(k in text for k in ['my refund', 'my money', 'charged']):
            return {
                "escalate": False,
                "escalation_reason": "General refund policy question; answerable with public documentation."
            }
        return {
            "escalate": True,
            "escalation_reason": "Refund, billing, or promotional credit issue requiring account/payment verification."
        }

    # Rule E: Item damage / Condition
    if is_damage_intent:
        if is_general_packaging_complaint:
            return {
                "escalate": False,
                "escalation_reason": "General product/packaging complaint without specific order lookup request."
            }
        return {
            "escalate": True,
            "escalation_reason": "Damaged/defective item resolution requiring order verification or replacement handling."
        }

    # Rule F: Marketplace / Third-party seller
    if has_marketplace:
        return {
            "escalate": True,
            "escalation_reason": "Marketplace seller dispute or A-to-z claim requiring internal intervention."
        }

    # Rule G: Order modification / Cancellation
    if has_cancellation_mod:
        if 'return policy' in text:
            return {
                "escalate": False,
                "escalation_reason": "Return policy inquiry; no private order cancellation required."
            }
        return {
            "escalate": True,
            "escalation_reason": "Order cancellation or address modification requiring internal account access."
        }

    # Rule H: Device support or Digital services with public troubleshooting
    if (is_device_intent or is_digital_intent) and has_public_troubleshooting:
        return {
            "escalate": False,
            "escalation_reason": "Public self-service device or app troubleshooting available; no private lookup needed."
        }

    # Rule I: Fallback check for general self-service / policy queries vs private lookup
    if any(k in text for k in ['how do i', 'what is the policy', 'how to', 'where can i find']) and not any(k in text for k in ['my order', 'my refund', 'my account']):
        return {
            "escalate": False,
            "escalation_reason": "General information request answerable with public self-service guidance."
        }

    # Default fallback for unspecified queries
    return {
        "escalate": False,
        "escalation_reason": "General query without actionable private order/account investigation."
    }
