# This is your main workspace.

# include your code and test it:
# execute both lines if you changed your code
include("src/project.jl")
include("test/runtests.jl")


# run the project
env_comp_MFG.plot_sol() # careful, it can take a while
