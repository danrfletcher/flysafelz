// The whole-UK box the map opens on. The same numbers as uk_bounds_lat_min, uk_bounds_lat_max, uk_bounds_lon_min and
// uk_bounds_lon_max in src/flysafelz/rules.yaml; tests/unit/test_ui_uk_bounds.py keeps the two equal.
export const UK_SOUTH = 49.8
export const UK_NORTH = 61
export const UK_WEST = -8.7
export const UK_EAST = 2

/** [[south, west], [north, east]], the order Leaflet's fitBounds takes. */
export const UK_BOUNDS: [[number, number], [number, number]] = [
  [UK_SOUTH, UK_WEST],
  [UK_NORTH, UK_EAST],
]
