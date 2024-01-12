from pdmproject.environment.wall import Wall


def test_adding_single_wall():
    wall = Wall((0, 0), (1, 1))
    assert wall.start_point == (0.0, 0.0)
