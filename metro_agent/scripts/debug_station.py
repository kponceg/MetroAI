from src.engine.engine import Engine
#This showed the attributes of Holder that are used in Engine
engine = Engine()

for _ in range(10):
    engine.increment_time(16)

print(dir(engine._components.stations[0]))