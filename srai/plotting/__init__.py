"""
This module contains plotting methods.

We provide some high-level plotting methods which work on the outputs of different `srai`
components. By default, `folium` based functions are exposed within `plotting` module. Additional
functions can be found in `srai.plotting.plotly_wrapper` module.
"""

from .folium_wrapper import plot_all_neighbourhood, plot_neighbours, plot_numeric_data, plot_regions

__all__ = ["plot_regions", "plot_numeric_data", "plot_neighbours", "plot_all_neighbourhood"]
