# Parameters
t_max = 600.0
J = 1.0
dt = 0.01

# Output directory
outdir = "./output"
if !isdir(outdir)
    mkpath(outdir)
end