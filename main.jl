using Flux, Plots, Images, FileIO, Statistics, Metalhead
using Random: rand
using MLUtils
using Pipe: @pipe

"A struct representing a single sample image consisting of a filename and label"
struct Sample{T}
    filename::String
    label::T
end

"Callable struct for extracting a one-hot label based on the subdirectory in which a sample image is located"
struct LabelFromDirectoryName
    classes::Vector{String}
end

"Callable which takes the path to a sample image and returns its label as a one-hot encoded vector based on the subdirectory name"
(l::LabelFromDirectoryName)(sample_filename::String) = @pipe split(sample_filename, "/")[end-1] |> Flux.onehot(_, l.classes)

"A struct which lazily reads sample images into memory on-demand"
struct LazyImageLoader
    samples::Vector{Sample}
    preprocessing_function::Function
end

"LazyImageLoader constructor which takes a list of sample files, a LabelFromDirectoryName struct, and a preprocessing function"
function LazyImageLoader(sample_files::Vector{String}, label::LabelFromDirectoryName, preprocessing)
    LazyImageLoader([Sample(filename, label(filename)) for filename in sample_files], preprocessing)
end

"Implement Base.length for LazyImageLoader"
function Base.length(v::LazyImageLoader)::Integer 
    return length(v.samples)
end

"Implement Base.getindex for LazyImageLoader when i is an int"
function Base.getindex(X::LazyImageLoader, i::Int)
    sample = X.samples[i]
    x = @pipe sample.filename |> load |> run_preprocessing(_, X.preprocessing_function) |> reshape(_, (size(_)..., 1))
    y = @pipe sample.label .|> Float32
    class = split(sample.filename, "/")[end-1]
    return x, y, class
end

"Implement Base.getindex for LazyImageLoader when i is a range"
function Base.getindex(X::LazyImageLoader, i::AbstractArray)
    xs = cat([X[j][1] for j in i]..., dims=(4))
    ys = hcat([X[j][2] for j in i]...)
    classes = [X[j][3] for j in i]
    return xs, ys, classes
end

"Run a preprocessing function over an image"
function run_preprocessing(img::Matrix{RGB{T}}, preprocessing) where {T <: Number}
    # @pipe img |> channelview |> rawview .|> Float32 |> permutedims(_, (3, 2, 1)) |> preprocessing
    @pipe img .|> float32 |> channelview |> permutedims(_, (3, 2, 1)) |> preprocessing
end

"Plot 16 randomly selected sample images"
function plot_samples(v::LazyImageLoader)
    sample_images = [@pipe sample.filename |> load(_) |> imresize(_, (256, 256)) for sample in rand(v.samples, 16)]
    plot([plot(x, axis=nothing, margin=0Plots.mm) for x in sample_images]..., layout=(4, 4), size=(256*4, 256*4))
    Plots.savefig("samples.png")
end

"Preprocess images by resizing to 256x256 and dividing all pixel intensities by 255"
function preprocess_image(img::Array{Float32, 3})
    @pipe img |> imresize(_, (256, 256))
end

"Load the sample data into a LazyImageLoader"
function load_data(data_directory::String)
    LazyImageLoader(list_samples(data_directory), LabelFromDirectoryName(["cat", "dog"]), preprocess_image)
end

"Get a list of filenames for all sample images"
function list_samples(data_directory::String)
    dirs = ["$data_directory/$d" for d in readdir(data_directory) if isdir("$data_directory/$d")]
    return ["$d/$filename" for d in dirs for filename in readdir(d)]
end

"Define the model"
function get_model()
    # Chain(ResNet34(pretrain=false), Dense(1000, 2), softmax) |> gpu
    Chain(
        Flux.Conv((3, 3), 3 => 32, relu, pad=SamePad()), 
        Flux.MaxPool((2, 2), pad=SamePad()), 
        Flux.Conv((3, 3), 32 => 64, relu, pad=SamePad()), 
        Flux.MaxPool((2, 2), pad=SamePad()), 
        Flux.Conv((3, 3), 64 => 128, relu, pad=SamePad()), 
        Flux.MaxPool((2, 2), pad=SamePad()), 
        Flux.Conv((3, 3), 128 => 256, relu, pad=SamePad()), 
        Flux.MaxPool((2, 2), pad=SamePad()), 
        Flux.flatten, 
        Flux.Dense(65536, 1024, relu),
        Flux.Dense(1024, 128, relu),
        Flux.Dense(128, 2),
        Flux.softmax, 
    )
end

"Main program loop"
function main_function()
    # Create Model
    model = get_model() |> gpu

    # Load Data
    data = load_data("data")
    data_train = @pipe data |> shuffleobs |> DataLoader(_, batchsize=32, shuffle=true)
    
    # Training Loop
    loss(x, y) = Flux.crossentropy(model(x), y, dims=1);
    parameters, opt = Flux.params(model), Flux.Optimise.ADAM(1e-8)
    for (i, (x, y, classes)) in enumerate(data_train)

        # Move Data To GPU
        x, y = gpu(x), gpu(y)

        # Log Loss And Accuracy
        l = @pipe loss(x, y) |> round(_, digits=6, base=10)
        accuracy = @pipe (Flux.onecold(model(x), 0:1) .== Flux.onecold(y, 0:1)) |> mean |> x -> x * 100 |> round |> Int
        println("Loss: $l, Accuracy: $accuracy%")

        # Print Progress
        total_iterations = length(data_train)
        println("Progress: $i / $total_iterations")

        # Compute Gradients
        grads = Flux.gradient(() -> loss(x, y), parameters)

        # Update Parameters
        Flux.Optimise.update!(opt, parameters, grads)
    end
end

"Program entry point"
main = main_function()
