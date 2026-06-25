from tac.loading.stl import load_stl
from tac.loading.tac import save_tac

if __name__ == "__main__":
    stl_parsed, state, welds, joints = load_stl("examples/gnoci/stl")
    save_tac(state, [welds, *joints], "examples/gnoci/gnoci.json")
