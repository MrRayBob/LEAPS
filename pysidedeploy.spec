[app]
title = LEAPS
project_dir = .
input_file = leaps/app.py
exec_directory = dist
project_file = pyproject.toml
icon = leaps/assets/leaps-app-icon.png

[python]
python_path =
packages = Nuitka

[qt]
qml_files =
excluded_qml_plugins =
modules = Core,Gui,Widgets,Svg
plugins = platforms,imageformats,styles

[nuitka]
macos.permissions =
mode = standalone
extra_args = --quiet --assume-yes-for-downloads --noinclude-qt-translations --include-data-dir=leaps/assets=leaps/assets --include-package=hops.pylightcurve41 --include-package=hops.thirdparty --include-package-data=hops --include-package-data=exoclock --include-package-data=exotethys
