from worker import Worker
from tools import arithmetic

worker1 = Worker("qwen3:1.7b")
worker1.respond("14+57")

# print(dir(arithmetic))