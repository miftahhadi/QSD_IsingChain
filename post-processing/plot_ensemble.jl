using JLD2
using CairoMakie

file1 = ARGS[1]
file2 = ARGS[2]
file3 = ARGS[3]

if isfile(file1)
    @info "Loading data from $file1"
    data1 = JLD2.load(file1)
else
    @error "File $file1 does not exist."
    exit(1)
end

if isfile(file2)
    @info "Loading data from $file2"
    data2 = JLD2.load(file2)
else
    @error "File $file2 does not exist."
    exit(1)
end

if isfile(file3)
    @info "Loading data from $file3"
    data3 = JLD2.load(file3)
else
    @error "File $file3 does not exist."
    exit(1)
end

println("Processing...")

x_ax = [log(t) for t in data1["times"]];

y1 = data1["mean_S"];
y2 = data2["mean_S"];
y3 = data3["mean_S"];

fig = Figure(size=(800, 600))
ax = Axis(fig[1,1], xlabel = "ln t", ylabel = "mean S(t)")

scatter!(ax, x_ax, y1, markersize=5, color=:orange, label="L = 64, N = 2000")
scatter!(ax, x_ax, y2, markersize=5, color=:blue, label="L = 128, N = 2000")
scatter!(ax, x_ax, y3, markersize=5, color=:blue, label="L = 192, N = 2000")
xlims!(ax, -7., 7.)

axislegend(ax; position = :lt)

savefile = "plot_ensemble_.png"

save(savefile, fig)

println("Plot saved to $savefile")