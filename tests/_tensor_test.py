import pytest

import nanotorch as torch


def test_item():
    assert torch.tensor(5.0).item() == 5.0


@pytest.mark.parametrize(
    ("test_input", "expected"),
    (
        (torch.tensor(7.0), 0),
        (torch.tensor([1.0]), 1),
        (torch.tensor([[1, 2], [3, 4]]), 2),
        (torch.tensor([[[1, 2, 3], [2, 3, 4]]]), 3),
    ),
)
def test_ndim(test_input, expected):
    assert test_input.ndim == expected


@pytest.mark.parametrize(
    ("test_input", "expected"),
    (
        (torch.tensor(5.0) + torch.tensor(3.0), torch.tensor(8.0)),
        (torch.tensor(5.0) - torch.tensor(3.0), torch.tensor(2.0)),
        (torch.tensor(5.0) * torch.tensor(3.0), torch.tensor(15.0)),
        (torch.tensor(3.0) / torch.tensor(3.0), torch.tensor(1.0)),
        (torch.tensor(17.0) // torch.tensor(-5.0), torch.tensor(-4)),
        (torch.tensor(5.0) ** torch.tensor(3.0), torch.tensor(125.0)),
        (torch.tensor(5.0) + 2, torch.tensor(7.0)),
        (torch.tensor(5.0) - 2, torch.tensor(3.0)),
        (torch.tensor(5.0) * 2, torch.tensor(10.0)),
        (2 + torch.tensor(3.0), torch.tensor(5.0)),
        (7 - torch.tensor(3.0), torch.tensor(4.0)),
        (2 * torch.tensor(3.0), torch.tensor(6.0)),
        (5 / torch.tensor(3.0), torch.tensor(1.6666666666666665)),
        (5 // torch.tensor(3.0), torch.tensor(1)),
        (2 ** torch.tensor(3.0), 8.0),
        (-torch.tensor(5.0), torch.tensor(-5.0)),
        (round(torch.tensor(5.667), 2), torch.tensor(5.67)),
    ),
)
def test_operations(test_input, expected):
    try:
        assert test_input.item() == expected.item()
    except AttributeError:
        assert test_input == expected


@pytest.mark.parametrize(
    ("test_input", "other", "expected"),
    (
        (torch.tensor(3.0), torch.tensor(2.0), 5.0),
        (torch.tensor(3.0), 3, 6.0),
        (3, torch.tensor(7.0), 10.0),
    ),
)
def test_add(test_input, other, expected):
    assert torch.add(test_input, other).item() == expected


@pytest.mark.parametrize(
    ("test_input", "other", "expected"),
    (
        (torch.tensor(3.0), torch.tensor(2.0), 1.0),
        (torch.tensor(3.0), 3, 0.0),
        (3, torch.tensor(7.0), -4.0),
    ),
)
def test_sub(test_input, other, expected):
    assert torch.sub(test_input, other).item() == expected


@pytest.mark.parametrize(
    ("test_input", "other", "expected"),
    (
        (torch.tensor(3.0), torch.tensor(2.0), 6.0),
        (torch.tensor(3.0), 3, 9.0),
        (3, torch.tensor(7.0), 21.0),
    ),
)
def test_mul(test_input, other, expected):
    assert torch.mul(test_input, other).item() == expected


@pytest.mark.parametrize(
    ("test_input", "other", "mode", "expected"),
    (
        (torch.tensor(3.0), torch.tensor(2.0), None, 1.5),
        (torch.tensor(3.0), 3, None, 1.0),
        (3, torch.tensor(1.0), None, 3.0),
        (torch.tensor(3.0), torch.tensor(2.0), "floor", 1),
        (torch.tensor(3.0), 3, "floor", 1),
        (3, torch.tensor(1.0), "floor", 3),
    ),
)
def test_div(test_input, other, mode, expected):
    assert torch.div(test_input, other, rounding_mode=mode).item() == expected


def test_zeros():
    assert torch.zeros().item() == 0.0


def test_ones():
    assert torch.ones().item() == 1.0
