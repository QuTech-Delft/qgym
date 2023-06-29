import shutil

import sphinx.cmd.build as sphinx_build
import sphinx.ext.apidoc as sphinx_apidoc

sphinx_apiddoc_args =[
    "-o=docs_build",
    "--templatedir=docs_files/templates",
    "--force",
    "--full",
    "--module-first",
    "--separate",
    "--implicit-namespaces",
    "qgym",
]

sphinx_build_args = ["docs_build","docs"]

sphinx_apidoc.main(sphinx_apiddoc_args)
sphinx_build.main(sphinx_build_args)

shutil.rmtree("docs_build")


