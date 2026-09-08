# UrbanFringePV

Code and author-generated spatial data supporting the manuscript
*Urban fringe solar photovoltaics synergise carbon mitigation and revegetation*.

## Contents

- `code/a01_boundary_task/` identifies urban areas associated with city
  locations and delineates 10-km urban-fringe zones.
- `code/a02_suitability_task/` applies land-cover, slope, solar-radiation,
  temperature and wind-speed constraints, excludes mapped existing PV and
  calculates suitable area.
- `code/config/` contains local paths, thresholds and workflow settings.
- `data/global_urban_fringe_pv_suitable_areas.tar.xz` contains the resulting
  suitable-area vector layer used in the country-level analysis.