using HDMJL
using Documenter

DocMeta.setdocmeta!(HDMJL, :DocTestSetup, :(using HDMJL); recursive=true)

makedocs(;
    modules=[HDMJL],
    authors="Rodrigo Grijalba, John Flores Rojas, Alexander Quispe, Anzony Quispe",
    repo="https://github.com/d2cml-ai/HDMJL.jl/blob/{commit}{path}#{line}",
    sitename="HDMJL.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://d2cml-ai.github.io/HDMJL.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/d2cml-ai/HDMJL.jl",
    devbranch="master",
)
