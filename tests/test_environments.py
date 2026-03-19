"""Tests for LoginFlow, CheckoutFlow, FormValidationFlow."""

import pytest
from causal_ui_gym import (
    CheckoutFlow,
    FormValidationFlow,
    LoginFlow,
    UIAction,
)


# ---------------------------------------------------------------------------
# LoginFlow
# ---------------------------------------------------------------------------

class TestLoginFlow:
    def test_initial_screen(self):
        task = LoginFlow()
        state = task.reset()
        assert state.screen == "login"
        assert not state.has_flag("logged_in")

    def test_happy_path_alice(self):
        task = LoginFlow()
        task.reset()
        task.step(UIAction("fill_username", {"value": "alice"}))
        task.step(UIAction("fill_password", {"value": "hunter2"}))
        result = task.step(UIAction("click_submit"))
        assert result.success
        assert result.done
        assert result.new_state.screen == "dashboard"
        assert result.new_state.has_flag("logged_in")

    def test_happy_path_bob(self):
        task = LoginFlow()
        task.reset()
        task.step(UIAction("fill_username", {"value": "bob"}))
        task.step(UIAction("fill_password", {"value": "passw0rd"}))
        result = task.step(UIAction("click_submit"))
        assert result.success

    def test_submit_without_username_fails(self):
        task = LoginFlow()
        task.reset()
        task.step(UIAction("fill_password", {"value": "hunter2"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success
        assert len(result.new_state.errors) > 0
        assert result.reward < 0

    def test_submit_without_password_fails(self):
        task = LoginFlow()
        task.reset()
        task.step(UIAction("fill_username", {"value": "alice"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success
        assert any("password" in e.lower() for e in result.new_state.errors)

    def test_wrong_credentials_fails(self):
        task = LoginFlow()
        task.reset()
        task.step(UIAction("fill_username", {"value": "alice"}))
        task.step(UIAction("fill_password", {"value": "wrongpass"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success
        assert any("invalid" in e.lower() for e in result.new_state.errors)
        # Password should be cleared after bad creds
        assert result.new_state.get_field("password") == ""

    def test_wrong_user_fails(self):
        task = LoginFlow()
        task.reset()
        task.step(UIAction("fill_username", {"value": "wrong_user"}))
        task.step(UIAction("fill_password", {"value": "hunter2"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success

    def test_submit_empty_form_fails(self):
        task = LoginFlow()
        task.reset()
        result = task.step(UIAction("click_submit"))
        assert not result.success
        assert result.reward < 0

    def test_available_actions_on_login_screen(self):
        task = LoginFlow()
        task.reset()
        names = {a.name for a in task.available_actions()}
        assert "fill_username" in names
        assert "fill_password" in names
        assert "click_submit" in names

    def test_steps_counted(self):
        task = LoginFlow()
        task.reset()
        assert task.steps == 0
        task.step(UIAction("fill_username", {"value": "alice"}))
        assert task.steps == 1

    def test_goal_description(self):
        task = LoginFlow()
        assert "log in" in task.goal.lower()

    def test_fill_field_positive_reward(self):
        task = LoginFlow()
        task.reset()
        result = task.step(UIAction("fill_username", {"value": "alice"}))
        assert result.reward > 0


# ---------------------------------------------------------------------------
# CheckoutFlow
# ---------------------------------------------------------------------------

class TestCheckoutFlow:
    def test_initial_screen(self):
        task = CheckoutFlow()
        state = task.reset()
        assert state.screen == "cart"
        assert not state.has_flag("order_placed")

    def test_happy_path(self):
        task = CheckoutFlow()
        task.reset()
        task.step(UIAction("add_item", {"item": "widget"}))
        task.step(UIAction("proceed_to_shipping"))
        task.step(UIAction("fill_address", {"value": "123 Main St"}))
        task.step(UIAction("proceed_to_payment"))
        task.step(UIAction("fill_card", {"value": "4111111111111111"}))
        task.step(UIAction("proceed_to_review"))
        result = task.step(UIAction("place_order"))
        assert result.success
        assert result.new_state.screen == "confirmation"
        assert result.new_state.has_flag("order_placed")

    def test_proceed_without_item_fails(self):
        task = CheckoutFlow()
        task.reset()
        result = task.step(UIAction("proceed_to_shipping"))
        assert not result.success
        assert len(result.new_state.errors) > 0
        assert result.new_state.screen == "cart"

    def test_proceed_to_payment_without_address_fails(self):
        task = CheckoutFlow()
        task.reset()
        task.step(UIAction("add_item", {"item": "widget"}))
        task.step(UIAction("proceed_to_shipping"))
        result = task.step(UIAction("proceed_to_payment"))
        assert not result.success
        assert result.new_state.screen == "shipping"
        assert any("address" in e.lower() for e in result.new_state.errors)

    def test_proceed_to_review_without_payment_fails(self):
        task = CheckoutFlow()
        task.reset()
        task.step(UIAction("add_item", {"item": "widget"}))
        task.step(UIAction("proceed_to_shipping"))
        task.step(UIAction("fill_address", {"value": "123 Main St"}))
        task.step(UIAction("proceed_to_payment"))
        result = task.step(UIAction("proceed_to_review"))
        # No card filled → should fail
        assert not result.success
        assert result.new_state.screen == "payment"

    def test_place_order_without_review_fails(self):
        """Trying place_order from cart is invalid."""
        task = CheckoutFlow()
        task.reset()
        # place_order is not available on cart screen
        available = {a.name for a in task.available_actions()}
        assert "place_order" not in available

    def test_bad_card_rejected(self):
        task = CheckoutFlow()
        task.reset()
        task.step(UIAction("add_item", {"item": "widget"}))
        task.step(UIAction("proceed_to_shipping"))
        task.step(UIAction("fill_address", {"value": "123 Main St"}))
        task.step(UIAction("proceed_to_payment"))
        task.step(UIAction("fill_card", {"value": "bad"}))
        result = task.step(UIAction("proceed_to_review"))
        assert not result.success
        assert any("card" in e.lower() for e in result.new_state.errors)

    def test_back_navigation_works(self):
        task = CheckoutFlow()
        task.reset()
        task.step(UIAction("add_item", {"item": "widget"}))
        task.step(UIAction("proceed_to_shipping"))
        result = task.step(UIAction("back_to_cart"))
        assert result.new_state.screen == "cart"

    def test_cart_filled_flag(self):
        task = CheckoutFlow()
        state = task.reset()
        assert not state.has_flag("cart_filled")
        result = task.step(UIAction("add_item", {"item": "widget"}))
        assert result.new_state.has_flag("cart_filled")


# ---------------------------------------------------------------------------
# FormValidationFlow
# ---------------------------------------------------------------------------

class TestFormValidationFlow:
    def test_initial_screen(self):
        task = FormValidationFlow()
        state = task.reset()
        assert state.screen == "form"
        assert not state.has_flag("form_submitted")

    def test_happy_path(self):
        task = FormValidationFlow()
        task.reset()
        task.step(UIAction("fill_email", {"value": "alice@example.com"}))
        task.step(UIAction("fill_age", {"value": "25"}))
        task.step(UIAction("fill_username", {"value": "alice"}))
        result = task.step(UIAction("click_submit"))
        assert result.success
        assert result.new_state.screen == "success"
        assert result.new_state.has_flag("form_submitted")

    def test_submit_empty_form_shows_all_errors(self):
        task = FormValidationFlow()
        task.reset()
        result = task.step(UIAction("click_submit"))
        assert not result.success
        assert len(result.new_state.errors) >= 3

    def test_invalid_email_rejected(self):
        task = FormValidationFlow()
        task.reset()
        task.step(UIAction("fill_email", {"value": "not-an-email"}))
        task.step(UIAction("fill_age", {"value": "25"}))
        task.step(UIAction("fill_username", {"value": "alice"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success
        assert any("@" in e or "email" in e.lower() for e in result.new_state.errors)

    def test_underage_rejected(self):
        task = FormValidationFlow()
        task.reset()
        task.step(UIAction("fill_email", {"value": "alice@example.com"}))
        task.step(UIAction("fill_age", {"value": "17"}))
        task.step(UIAction("fill_username", {"value": "alice"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success
        assert any("18" in e for e in result.new_state.errors)

    def test_non_numeric_age_rejected(self):
        task = FormValidationFlow()
        task.reset()
        task.step(UIAction("fill_email", {"value": "alice@example.com"}))
        task.step(UIAction("fill_age", {"value": "abc"}))
        task.step(UIAction("fill_username", {"value": "alice"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success

    def test_short_username_rejected(self):
        task = FormValidationFlow()
        task.reset()
        task.step(UIAction("fill_email", {"value": "alice@example.com"}))
        task.step(UIAction("fill_age", {"value": "25"}))
        task.step(UIAction("fill_username", {"value": "ab"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success

    def test_special_char_username_rejected(self):
        task = FormValidationFlow()
        task.reset()
        task.step(UIAction("fill_email", {"value": "alice@example.com"}))
        task.step(UIAction("fill_age", {"value": "25"}))
        task.step(UIAction("fill_username", {"value": "alice!"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success

    def test_partial_fill_errors(self):
        """Only missing username triggers one error."""
        task = FormValidationFlow()
        task.reset()
        task.step(UIAction("fill_email", {"value": "alice@example.com"}))
        task.step(UIAction("fill_age", {"value": "25"}))
        result = task.step(UIAction("click_submit"))
        assert not result.success
        assert len(result.new_state.errors) == 1

    def test_filling_valid_field_gives_positive_reward(self):
        task = FormValidationFlow()
        task.reset()
        result = task.step(UIAction("fill_email", {"value": "alice@example.com"}))
        assert result.reward > 0

    def test_filling_invalid_value_gives_negative_reward(self):
        task = FormValidationFlow()
        task.reset()
        result = task.step(UIAction("fill_email", {"value": "bad"}))
        assert result.reward < 0
