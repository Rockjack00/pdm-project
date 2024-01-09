import matplotlib.pyplot as plt

from pdmproject.environment import GateWall, PDMWorldCreator, PerimeterWall, Wall

world_plan = PDMWorldCreator()
world_plan.register(PerimeterWall((0, 0), 5, 7))
world_plan.register(GateWall((-0.25, 0), (2.5, 0), (1.25, 0)))
world_plan.register(Wall((-1, 1), (2, -3)))
world_plan.register(Wall((-2.5, 3), (2.5, 3)))
world_plan.register(
    GateWall((-1, 1), (-2.5, 1), (-1.75, 1), extra_data={"rgba": [0.9, 0.4, 0.4, 1.0]})
)

plt.switch_backend("TkAgg")

# world_plan

fig, ax = world_plan.plot2d()
plt.show()