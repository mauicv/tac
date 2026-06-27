# TAC

Tac is a simple XPBD physics engine that does 2 things:

1. State machine for state estimation from partial information
2. Learnable Robot physics prior for world model learning.

## TODO:

- [x] Implement State:
    - class State: 
    - initiallised with (x,y,z) tensor
    - contains x_current and x_previous torch tensors
    - points will be indexd by there integer_ids (and potentially also assigned names)
- [x] Implement Length constraint:
    - class Length: 
    - init with (int i, int j, float alpha)
    - computes length between points and stores it
    - apply method: applies XPBD contsraint resolution
- [x] Implement Weld:
    - class Weld:
    - init with List[int]. require len(List[int])>=3
    - creates list of Length constraints for each point
- [x] Implement FacetedWeld
- [x] Implement Render
- [x] Implement Joint
    - class Joint:
    - takes axis_points, from_points, to_points
- [x] Implement IMU constraint
- [x] Implement examples:
    - [x] Box 
    - [x] Joint
    - [x] Line
    - [x] IMU
    - [ ] Gnoci
        - [ ] add COM point to each stl object and use length set
        - [ ] Use COM points as joint from and tos
- [x] Implement vectorized length and joint constraints
- [ ] Implement vectorized contact constraint
- [ ] implement colour sorting for vectorization