module CatsAndDogsModels

export pretrained_model, basic_model

using Flux, Metalhead

function pretrained_model()
    return Flux.Chain(Metalhead.ResNet(34, pretrain=true), Flux.Dense(1000, 2), Flux.softmax)
end

function untrained_model()
    return Flux.Chain(Metalhead.ResNet(34, pretrain=false), Flux.Dense(1000, 2), Flux.softmax)
end

function basic_model()
    Chain(
        Flux.Conv((3, 3), 3 => 32, relu), 
        Flux.MaxPool((2, 2)), 
        Flux.Conv((3, 3), 32 => 64, relu), 
        Flux.MaxPool((2, 2)), 
        Flux.Conv((3, 3), 64 => 128, relu), 
        Flux.MaxPool((2, 2)), 
        Flux.Conv((3, 3), 128 => 256, relu), 
        Flux.MaxPool((2, 2)), 
        Flux.flatten, 
        Flux.Dense(50176, 1024, relu),
        Flux.Dense(1024, 128, relu),
        Flux.Dense(128, 2),
        Flux.softmax
    )
end


end