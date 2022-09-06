using HDMjl
using Documenter

DocMeta.setdocmeta!(HDMjl, :DocTestSetup, :(using HDMjl); recursive=true)

makedocs(;
    modules=[HDMjl],
    authors="Jhon Flores Rojas, Rodrigo Grijalba, Alexander Quispe, Anzony Quispe",
    repo="https://github.com/d2cmjl-ai/HDMjl.jl/blob/{commit}{path}#{line}",
    sitename="HDMjl.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://d2cmjl-ai.github.io/HDMjl.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/d2cmjl-ai/HDMjl.jl",
    devbranch="main",
)
