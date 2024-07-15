plugin-template
===============
_Biological Data Science Summer School, 7-20 July 2024, Uzhhorod, Ukraine._

plugin-template: napari plugin template for BDS^3 2024

Based on official [naparu Plugins Guide](https://napari.org/stable/plugins/first_plugin.html)

### Plugin structure:
```
plugin-template/
└── src/
│   └── plugin_template/
│       ├── __init__.py
│       ├── napari.yaml
│       └── _widget.py
├─── pyproject.toml
├─── setup.cfg
├─── README.md
├─── LICENSE
└─── .gitignore
```

### Dependency
- python >= 3.8
- matplotlib
- numpy
- scikit-image

### Local installation in editable mode with `pip`:
```
python -m pip install -e .
```

### Widgets
- Widget demo
- Plot demo
- Threading demo