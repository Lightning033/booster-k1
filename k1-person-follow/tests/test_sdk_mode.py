"""Reading the robot's mode out of whatever shape GetMode returns.

The field name differs between SDK builds. Reading it wrongly does not
just mis-print: is_walking() then reads False and ensure_walking() refuses
to start the follow loop while the robot is in fact walking.
"""

import unittest

from support import PROJECT_ROOT  # noqa: F401

from sdk_client.sdk_client import _mode_field, is_walking



class ModeResponseWithUnderscore:
    """This SDK build: struct members carry a trailing underscore."""
    def __init__(self, value="kWalking"):
        self.mode_ = value


class ModeResponsePlain:
    def __init__(self, value="kWalking"):
        self.mode = value


class ModeResponseOpaque:
    """No mode field at all - the shape that caused the bug."""
    pass


class ModeResponseEnumOnly:
    def __init__(self):
        self.state_ = "Mode.kPrepare"


class TestModeField(unittest.TestCase):
    def test_trailing_underscore_is_read(self):
        self.assertEqual(_mode_field(ModeResponseWithUnderscore()), "kWalking")

    def test_plain_name_still_works(self):
        self.assertEqual(_mode_field(ModeResponsePlain()), "kWalking")

    def test_a_bare_string_passes_through(self):
        self.assertEqual(_mode_field("kWalking"), "kWalking")

    def test_an_opaque_response_is_unknown_not_a_repr(self):
        """The actual bug: it must not return the object itself."""
        result = _mode_field(ModeResponseOpaque())
        self.assertIsNone(result, "returned %r instead of None" % (result,))

    def test_an_enum_under_another_name_is_found(self):
        self.assertEqual(str(_mode_field(ModeResponseEnumOnly())),
                         "Mode.kPrepare")

    def test_none_in_none_out(self):
        self.assertIsNone(_mode_field(None))


class TestIsWalking(unittest.TestCase):
    def test_walking_variants(self):
        for text in ("kWalking", "Mode.kWalking", "WALKING", "kwalking"):
            self.assertTrue(is_walking(text), "%r should read as walking" % text)

    def test_non_walking_variants(self):
        for text in ("kPrepare", "kDamping", "Mode.kCustom"):
            self.assertFalse(is_walking(text), "%r should not read as walking" % text)

    def test_the_old_repr_would_have_read_false(self):
        """Proof of the failure: the repr contains no mode name."""
        repr_text = "<booster_robotics_sdk_python._core.GetModeResponse object at 0x78fde825e870>"
        self.assertFalse(is_walking(repr_text),
                         "this is exactly why ensure_walking refused to start")
