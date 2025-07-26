"""Test case base class."""
import sys

from absl import flags
from absl import logging
from absl.testing import absltest

FLAGS = flags.FLAGS


class TestCase(absltest.TestCase):
    """Base class for test cases."""

    @classmethod
    def setUpClass(cls):
        """Sets up the test environment."""
        super().setUpClass()
        if not FLAGS.is_parsed():
            print("Setting up absl.")
            # Running from VSCode unittest runner.
            FLAGS([sys.argv[0]])
            # Set up absl logging.
            logging.set_verbosity(logging.INFO)
            logging.set_stderrthreshold('info')
