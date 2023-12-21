from pdmproject.sampling import SimpleSampler


def test_simple_sampler():
    sampler = SimpleSampler()

    lower = sampler._lower_bound
    upper = sampler._upper_bound

    sample = sampler.get_sample()

    assert (lower <= sample).all()
    assert (upper > sample).all()
