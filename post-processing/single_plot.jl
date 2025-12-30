using JLD2
using CairoMakie

# Get the filename
println("Input the JLD2 filepath:")
file1 = readline()
println("You entered: $file1")

if isfile(file1)
    @info "Loading data from $file1"
    data1 = JLD2.load(file1)
else
    @error "File $file1 does not exist."
    exit(1)
end

println("Processing...")

x_ax = [log(t) for t in data1["times"]];

y1 = data1["mean_S"];

fig = Figure(size=(800, 600))
ax = Axis(fig[1,1], xlabel = "ln t", ylabel = "mean S(t)")

scatter!(ax, x_ax, y1, markersize=5, color=:orange, label="L = 64, N = 2000")
xlims!(ax, -7., 7.)

axislegend(ax; position = :lt)

savefile = "single_plot.png"

save(savefile, fig)

println("Plot saved to $savefile")