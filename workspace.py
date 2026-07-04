import jax
import jax.random as random

key = random.PRNGKey(0)

r, key = random.split(key, 2)
a = random.randint(r, (4, 10, 2), 0, 10)

b = a.reshape(4, -1)
print(b)
c = b.bincount(minlength=10)
print(c)