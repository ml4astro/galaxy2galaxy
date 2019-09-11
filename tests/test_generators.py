import pytest
import galaxy2galaxy.problems as problems

def test_problems():
    problem_list = problems.available()
    assert 'img2img_hsc' in problem_list
