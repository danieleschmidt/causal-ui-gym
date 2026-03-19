"""
CheckoutFlow — causal UI environment modelling an e-commerce checkout funnel.

Causal structure (screens must be visited in order)
---------------------------------------------------
  cart → shipping → payment → review → confirmation

Causal side-effects enforced:
- Can't go to shipping without items in cart
- Can't go to payment without a shipping address
- Can't go to review without payment details
- Clicking "place order" on review → confirmation + order_placed flag
- Skipping any step results in an error and no navigation

Optimal path: add_item → proceed_to_shipping → fill_address →
              proceed_to_payment → fill_card → proceed_to_review →
              place_order   (7 steps)
"""

from __future__ import annotations

from typing import List

from ..core import (
    ActionResult,
    CausalUIEnvironment,
    CausalUITask,
    UIAction,
    UIState,
)

_OPTIMAL_STEPS = 7


class _CheckoutEnv(CausalUIEnvironment):
    name = "CheckoutFlow"
    max_steps = 25

    def _initial_state(self) -> UIState:
        return UIState(
            screen="cart",
            fields={
                "shipping_address": "",
                "card_number": "",
            },
            flags={
                "cart_filled": False,
                "shipping_done": False,
                "payment_done": False,
                "order_placed": False,
            },
            errors=[],
            metadata={"optimal_steps": _OPTIMAL_STEPS},
        )

    def _available_actions(self, state: UIState) -> List[UIAction]:
        screen = state.screen
        actions: List[UIAction] = []

        if screen == "cart":
            actions += [
                UIAction("add_item", {"item": "widget"}),
                UIAction("proceed_to_shipping"),
            ]

        elif screen == "shipping":
            actions += [
                UIAction("fill_address", {"value": "123 Main St"}),
                UIAction("fill_address", {"value": ""}),  # empty (bad)
                UIAction("proceed_to_payment"),
                UIAction("back_to_cart"),
            ]

        elif screen == "payment":
            actions += [
                UIAction("fill_card", {"value": "4111111111111111"}),
                UIAction("fill_card", {"value": "bad"}),
                UIAction("proceed_to_review"),
                UIAction("back_to_shipping"),
            ]

        elif screen == "review":
            actions += [
                UIAction("place_order"),
                UIAction("back_to_payment"),
            ]

        elif screen == "confirmation":
            actions += [UIAction("go_home")]

        return actions

    def _transition(self, state: UIState, action: UIAction) -> ActionResult:
        state = state.record_action(action.name)
        state = state.clear_errors()

        # --- cart ---
        if action.name == "add_item":
            s = state.with_flag("cart_filled", True)
            return ActionResult(new_state=s, reward=0.2)

        if action.name == "proceed_to_shipping":
            if not state.has_flag("cart_filled"):
                s = state.with_error("Your cart is empty. Add items before checking out.")
                return ActionResult(new_state=s, reward=-0.5)
            s = state.with_screen("shipping")
            return ActionResult(new_state=s, reward=0.2)

        if action.name == "back_to_cart":
            return ActionResult(new_state=state.with_screen("cart"), reward=0.0)

        # --- shipping ---
        if action.name == "fill_address":
            value = action.payload.get("value", "")
            s = state.with_field("shipping_address", value)
            return ActionResult(new_state=s, reward=0.1 if value else -0.1)

        if action.name == "proceed_to_payment":
            if not state.has_flag("cart_filled"):
                s = state.with_error("Cannot proceed: cart is empty.")
                return ActionResult(new_state=s, reward=-0.5)
            addr = state.get_field("shipping_address")
            if not addr:
                s = state.with_error("Please enter a shipping address.")
                return ActionResult(new_state=s, reward=-0.5)
            s = state.with_screen("payment").with_flag("shipping_done", True)
            return ActionResult(new_state=s, reward=0.2)

        if action.name == "back_to_shipping":
            return ActionResult(new_state=state.with_screen("shipping"), reward=0.0)

        # --- payment ---
        if action.name == "fill_card":
            value = action.payload.get("value", "")
            s = state.with_field("card_number", value)
            return ActionResult(new_state=s, reward=0.1 if len(value) >= 13 else -0.1)

        if action.name == "proceed_to_review":
            if not state.has_flag("shipping_done"):
                s = state.with_error("Complete shipping info first.")
                return ActionResult(new_state=s, reward=-0.5)
            card = state.get_field("card_number")
            if len(card) < 13:
                s = state.with_error("Invalid card number.")
                return ActionResult(new_state=s, reward=-0.5)
            s = state.with_screen("review").with_flag("payment_done", True)
            return ActionResult(new_state=s, reward=0.2)

        if action.name == "back_to_payment":
            return ActionResult(new_state=state.with_screen("payment"), reward=0.0)

        # --- review ---
        if action.name == "place_order":
            if not state.has_flag("payment_done"):
                s = state.with_error("Cannot place order: payment incomplete.")
                return ActionResult(new_state=s, reward=-0.5)
            s = (
                state
                .with_screen("confirmation")
                .with_flag("order_placed", True)
            )
            return ActionResult(new_state=s, reward=1.0, done=True, success=True,
                                info={"optimal_steps": _OPTIMAL_STEPS})

        # --- confirmation ---
        if action.name == "go_home":
            return ActionResult(new_state=state.with_screen("cart"), reward=0.0)

        return ActionResult(new_state=state, reward=-0.1,
                            info={"reason": "unhandled_action"})


def CheckoutFlow() -> CausalUITask:
    """
    Return a CausalUITask wrapping the checkout environment.

    Goal: reach the confirmation screen with order_placed=True.
    """
    env = _CheckoutEnv()
    return CausalUITask(
        env=env,
        goal="Complete checkout and place the order.",
        is_goal=lambda s: s.screen == "confirmation" and s.has_flag("order_placed"),
    )
