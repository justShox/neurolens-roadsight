# Scene description (own notes; replaces the missing samples/camera.md)

## Layout
- Four-way (plus-shaped) intersection; fixed CCTV looking along one approach.
- Geometry lives in `scene/scene.json` (normalized coords) and is warped to each clip via SIFT+RANSAC against `scene/reference.jpg`.
- Lanes: `lane1`–`lane10`, plus `bus_road`, `turn_lane`, `main`, `left` / `right` roads and exits (`left_exit`, `forward_exit`, `right_exit_slip`).
- Solid lane dividers: `solid1`–`solid4`. Stop / yield geometry and turn restrictions encoded as polygons/polylines in the same file (stop line, no-U-turn / slip zones used by the rules).
- Crosswalks: `near_crosswalk`, `left_crosswalk`, `right_crosswalk`, plus `crosswalk_illegal` for off-crossing pedestrian zones.
- Islands: `island_1`, `island_2`; side features: `driveway`, `parking_zone`.

## Traffic signal
- The signal for the camera-side approach faces away from the camera and is not readable.
- The signal of the opposite approach faces the camera and is readable (crop boxes in `scene.json`).
- Opposite approaches switch in sync (checked on the sample videos), so the opposite signal is used as the signal state for both.
- Fallback when the opposite signal is occluded: cross traffic moving means red for our approach.
- Yellow is not treated as red.

## Labelling conventions agreed by the team
- red_light: the vehicle front crosses the stop line while the signal is already red.
- stop_line: a vehicle stands past the stop line while the signal is red and does not enter the intersection,
  regardless of when it crossed the line (including on yellow).
  Start = max(moment it stops, moment red turns on). End = moment green turns on.
  Counts only if the vehicle front is clearly past the line; borderline cases get a note.
- A vehicle that crosses the stop line on yellow and continues through the intersection is not an event.
- A vehicle standing on the crossing while a pedestrian is on it is failure_to_yield (separate from stop_line).
- congestion: all lanes of one direction stand still or crawl. A normal red-light queue that clears on green
  is not congestion; it counts when the queue does not clear on green or lasts longer than one signal cycle.
  Start = queue stops moving, end = queue clears. Minimum 10 s. Both directions at once = one segment.
  Spillback (queue blocked by a jam beyond the intersection while green) counts; mostly-red cases get a note.
- solid_line_crossing is only the lane-change manoeuvre across a solid line (wheel crosses -> fully in new lane,
  usually 1–3 s). A vehicle standing on a line is not an event.
- Same-class events are merged into one segment when they overlap or the gap between them is under 2 s;
  a gap of 2 s or more starts a new segment.
