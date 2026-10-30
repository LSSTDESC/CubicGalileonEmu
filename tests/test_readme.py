"""Run the Python examples in README.md so the documentation cannot drift from the code."""

import os
import re

import pytest

README = os.path.join(os.path.dirname(__file__), '..', 'README.md')
BLOCKS = re.findall(r'```python\n(.*?)```', open(README).read(), flags=re.S)


def test_readme_has_examples():
    assert BLOCKS


@pytest.mark.parametrize('block', BLOCKS, ids=['block%d' % i for i in range(len(BLOCKS))])
def test_readme_example_runs(block):
    exec(compile(block, 'README.md', 'exec'), {})
