module CatsAndDogs
export Sample, LabelFromDirectoryName, LabelFromFilename, LazyImageLoader, StrictImageLoader, plot_samples

using Flux, Plots, Images, FileIO, Statistics, Metalhead
using Random: rand
using MLUtils
using Pipe: @pipe

struct Sample{T}
    filename::String
    label::T
end

struct LabelFromDirectoryName
    classes::Vector{String}
end

(l::LabelFromDirectoryName)(sample_filename::String) = @pipe split(sample_filename, "/")[end-1] |> Flux.onehot(_, l.classes)

struct LabelFromFilename
    classes::Vector{String}
end

function (l::LabelFromFilename)(sample_filename::String)
    filename = split(sample_filename, "/")[end]
    class = filter(class -> contains(filename, class), l.classes)[1]
    return Flux.onehot(class, l.classes)
end

struct LabelFromDictionary{T}
    d::Dict{String,T}
end

(l::LabelFromDictionary)(sample_filename::String) = l.d[sample_filename]

struct LazyImageLoader
    samples::Vector{Sample}
    preprocessing_function::Function
end

function LazyImageLoader(sample_files::Vector{String}, label::Union{LabelFromDirectoryName, LabelFromFilename}, preprocessing)
    LazyImageLoader([Sample(filename, label(filename)) for filename in sample_files], preprocessing)
end

struct StrictImageLoader
    samples::Vector{Sample}
    features::Dict{String,Matrix{RGB{N0f8}}}
    preprocessing_function
end

function StrictImageLoader(sample_files::Vector{String}, label::Union{LabelFromDirectoryName, LabelFromFilename}, preprocessing)
    d = Dict{String,Matrix{RGB{N0f8}}}()
    for filename in sample_files
        println(filename)
        d[filename] = filename |> load
    end
    return StrictImageLoader([Sample(filename, label(filename)) for filename in sample_files], d, preprocessing)
end

function Base.length(v::Union{LazyImageLoader, StrictImageLoader})::Integer 
    return length(v.samples)
end

function Base.getindex(X::LazyImageLoader, i::Int)
    sample = X.samples[i]
    # x = @pipe sample.filename |> load |> run_preprocessing(_, X.preprocessing_function) |> reshape(_, (size(_)..., 1))
    x = @pipe sample.filename |> load |> run_preprocessing(_, X.preprocessing_function) |> reshape(_, (size(_)..., 1))
    y = @pipe sample.label |> [e for e in _]
    return x, y
end

function Base.getindex(X::StrictImageLoader, i::Int)
    sample = X.samples[i]
    x = @pipe X.features[sample.filename] |> run_preprocessing(_, X.preprocessing_function) |> reshape(_, (size(_)..., 1))
    y = @pipe sample.label |> [e for e in _]
    return x, y
end

function Base.getindex(X::Union{LazyImageLoader, StrictImageLoader}, i::AbstractArray)
    xs = cat([X[j][1] for j in i]..., dims=(4))
    ys = [X[j][2] for j in i]
    ys = hcat([X[j][2] for j in i]...)
    return xs, ys
end

function plot_samples(v::Union{LazyImageLoader, StrictImageLoader})
    sample_images = [@pipe sample.filename |> load(_) |> imresize(_, (256, 256)) for sample in rand(v.samples, 16)]
    plot([plot(x, axis=nothing, margin=0Plots.mm) for x in sample_images]..., layout=(4, 4), size=(256*4, 256*4))
    Plots.savefig("myplot.png")
end

function run_preprocessing(img::Matrix{RGB{T}}, preprocessing) where {T <: Number}
    @pipe img |> channelview |> rawview .|> Float32 |> permutedims(_, (3, 2, 1)) |> preprocessing
end

function apply_random(f, img, probability::Real)
    rand() <= probability ? f(img) : img
end

function apply_augmentations(img, augs::Vector{Tuple{Function,Float32}})
    for (f, likelihood) in augs
        img = apply_random(f, img, likelihood)
    end
    return img
end

function preprocess_image(img::Array{Float32, 3})
    # @pipe img |> imresize(_, (256, 256)) |> mapslices(MLUtils.normalise, _, dims=(1, 2))
    @pipe img |> imresize(_, (256, 256)) |> x -> x / Float32(255.0)
    # @pipe img |> imresize(_, (256, 256))
end

function load_data(data_directory::String)
    LazyImageLoader(list_samples(data_directory), LabelFromFilename(["cat", "dog"]), preprocess_image)
end

function list_samples(data_directory::String)
    dirs = ["$data_directory/$d" for d in readdir(data_directory) if isdir("$data_directory/$d")]
    ["$d/$filename" for d in dirs for filename in readdir(d)]
end

function get_model()
    # return Chain(ResNet34(pretrain=false), Dense(1000, 2), softmax)
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
    )
end

function define_loss(model)
    loss(x, y) = @pipe x |> model |> Flux.crossentropy(_, y, dims=2);
    return loss
end

function evaluate_helper(x, y, model) 
    prediction = @pipe x |> model |> Flux.onecold(_, 0:1)
    labels = Flux.onecold(y, 0:1)
    return labels .== prediction
end

function evaluate(dataset, m)
	[evaluate_helper(x, y, m) for (x, y) in dataset] |>
	Iterators.flatten |>
	mean
end

function get_callback(data, model)
    function progress()
        evaluate(data, model) |> println
    end
    return progress
end

function main_function()
    model = get_model()
    data = load_data("data") |> shuffleobs
    data_train = DataLoader(data, batchsize=8, shuffle=true)
    
    # Training Loop
    parameters = Flux.params(model)
    # opt = Flux.Optimise.ADAM(0.00001)
    opt = Flux.Optimise.ADAM(0.1)
    loss(x, y) = Flux.logitcrossentropy(model(x), y, dims=1);
    for (x, y) in data_train
        x_type = typeof(x)
        x_size = size(x)
        y_type = typeof(y)
        y_size = size(y)
        l = loss(x, y)
        # "X Type: $x_type\nX Size: $x_size\nY Type: $y_type\nY Size: $y_size\nLoss: $l" |> println
        # "--------------------" |> println
        l |> println
        grads = Flux.gradient(() -> loss(x, y), parameters)
        Flux.Optimise.update!(opt, parameters, grads)
    end
end


end
