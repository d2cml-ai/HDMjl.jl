### Instrucciones para crear el pdf

# - Tener `pandoc`
# - Tener `Latex`
# - Instalar los siguientes paquetes `] add Weave Highlights`
# - Ir a la carpeta principal (meidante julia, bash o VSCode)
#     - asegurarse que se esta trabajando con el entorno general β`] activate`
# - correr en el bash o la terminal de vscode `julia test/hdm_build.jl`

########## Code 

using Weave, Highlights

# @time weave("hdm.jmd"; doctype = "md2pdf", highlight_theme = Highlights.Themes.MonokaiMiniTheme, pandoc_options = ["--toc", "--toc-depth= 3", "--number-sections", "--self-contained"])
# @time weave("hdm.jmd"; doctype="pandoc2pdf", pandoc_options=["--toc", "--toc-depth= 1", "--number-sections", "--self-contained"], highlight_theme = Highlights.Themes.TangoTheme)
@time weave("tutorial/hdm.jmd"; doctype="md2tex", pandoc_options=["--toc", "--toc-depth= 1", "--number-sections", "--self-contained"])





