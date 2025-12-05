#!/usr/bin/env python3
"""
Rough 1D current + trip-time model for the St. Johns River (FL)
between Green Cove Springs and the mouth.

- Nodes: NOAA tidal current prediction stations (or proxies).
- Currents: pulled in real time from NOAA CO-OPS Data API
  using product=currents_predictions.
- Between nodes: width-weighted interpolation of nodal currents.
- Boat: constant speed through the water (knots).
- Output: estimated trip duration given a start time and direction.

You *must* edit the NODE_CONFIG section with realistic distances,
widths, and station IDs that support `currents_predictions`.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, UTC

import requests

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------

API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# Maximum hours of predictions to request (NOAA limit: 1 month for non-max/slack)
MAX_HOURS_AHEAD = 31 * 24  # 744 hours

# River nodes ordered from MOUTH (downstream) to UPSTREAM.
# Distances are along the channel in kilometers (you should refine).
# Widths are rough effective channel widths in meters (tune from charts/experience).
#
# station_id must be a valid NOAA currents-prediction station ID, e.g. 'SJR9801'
# or one of the subordinate stations referenced to it. You can look these up on:
#   https://tidesandcurrents.noaa.gov/noaacurrents/
#
# sign_downstream:
#   +1 if positive currents from NOAA mean "downstream" (toward the ocean),
#   -1 if positive means "upstream". You may need to flip after inspecting data.
@dataclass
class RiverNode:
    name: str
    station_id: str
    channel_km: float
    width_m: float
    sign_downstream: int = 1
    bin: int = 0  # 0 -> default bin (near surface) for predictions


NODE_CONFIG: list[RiverNode] = [
    # NOTE: These station IDs are placeholders / examples.
    # Replace with actual St. Johns current prediction station IDs you want.
    RiverNode(
        name="Mouth (St Johns River Entrance)",
        station_id="SJR9801",  # example ref station at entrance
        channel_km=0.0,
        width_m=1200.0,
        sign_downstream=1,
    ),
    RiverNode(
        name="Jacksonville / Commodore Point",
        station_id="ACT8036_1",  # example subordinate station
        channel_km=25.0,        # rough along-channel distance, tweak this
        width_m=800.0,
        sign_downstream=1,
    ),
    RiverNode(
        name="Green Cove Springs (approx)",
        station_id="ACT8XXX_1",  # TODO: replace with a suitable station or proxy
        channel_km=70.0,        # rough; refine based on charts
        width_m=500.0,
        sign_downstream=1,
    ),
]

# Boat properties
BOAT_SPEED_KTS = 5.0  # constant speed through water in knots (edit this)

# Temporal resolution of current predictions (minutes).
# Allowed values per NOAA docs: 1, 6, 10, 30, 60, or 'h' for hourly. We pick 10.
PREDICTION_INTERVAL_MIN = 10

# Time zone choice. 'gmt' is simplest; NOAA also supports 'lst_ldt'.
NOAA_TIME_ZONE = "gmt"

# --------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------

def knots_to_km_per_hour(kts: float) -> float:
    return kts * 1.852  # exact definition

def km_per_hour_to_km_per_sec(kmh: float) -> float:
    return kmh / 3600.0

def parse_noaa_timestamp(ts: str) -> datetime:
    """
    NOAA currents_predictions time string is ISO-like; treat as naive in the
    requested time_zone. For 'gmt' we attach UTC tzinfo.
    """
    # Examples are typically "2025-12-04 06:00"
    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
    if NOAA_TIME_ZONE == "gmt":
        return dt.replace(tzinfo=UTC)
    else:
        # For simplicity, leave as naive; caller must interpret.
        return dt

def fetch_currents_predictions(
    station_id: str,
    start: datetime,
    hours: int,
    interval_minutes: int = PREDICTION_INTERVAL_MIN,
    bin_number: int = 0,
) -> list[tuple[datetime, float]]:
    """
    Fetch tidal current predictions from NOAA for a given station and time window.

    Returns a list of (timestamp, current_speed_knots_signed) sorted by time.

    We use:
      product=currents_predictions
      begin_date=YYYYMMDD
      range=hours
      interval={interval_minutes}
      units=english
      time_zone=NOAA_TIME_ZONE
      bin=bin_number
      format=json
    """
    if hours > MAX_HOURS_AHEAD:
        raise ValueError(f"hours={hours} exceeds NOAA month-long limit ({MAX_HOURS_AHEAD})")

    # NOAA's begin_date is date-only; we request from that date onward for "range" hours.
    # Then we sub-select times >= the actual start time in our integration.
    begin_date_str = start.strftime("%Y%m%d")

    params = {
        "station": station_id,
        "product": "currents_predictions",
        "begin_date": begin_date_str,
        "range": str(hours),
        "time_zone": NOAA_TIME_ZONE,
        "interval": str(interval_minutes),
        "units": "english",
        "format": "json",
        "application": "stjohns_1d_example",
    }
    if bin_number != 0:
        params["bin"] = str(bin_number)

    resp = requests.get(API_URL, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(f"NOAA API error for station {station_id}: {data['error']}")

    preds = data.get("data", [])
    out: list[tuple[datetime, float]] = []

    for row in preds:
        # Default mode returns value 'v' in knots (signed along major axis, flood/ebb).
        # See Data API responseHelp for exact fields.
        t = parse_noaa_timestamp(row["t"])
        v_knots = float(row["v"])
        out.append((t, v_knots))

    # Sort and return
    out.sort(key=lambda x: x[0])
    return out

def build_node_time_series(
    nodes: list[RiverNode],
    start: datetime,
    hours: int,
) -> dict[str, list[tuple[datetime, float]]]:
    """
    Fetch currents_predictions for all nodes and return dict:
    station_id -> [(time, signed_current_knots_downstream), ...]
    """
    series: dict[str, list[tuple[datetime, float]]] = {}
    for node in nodes:
        raw = fetch_currents_predictions(
            node.station_id,
            start=start,
            hours=hours,
            interval_minutes=PREDICTION_INTERVAL_MIN,
            bin_number=node.bin,
        )
        # Apply sign convention so positive = downstream current
        adjusted = [
            (t, node.sign_downstream * v) for (t, v) in raw if t >= start
        ]
        series[node.station_id] = adjusted
    return series

def reindex_to_common_times(
    node_series: dict[str, list[tuple[datetime, float]]]
) -> tuple[list[datetime], dict[str, list[float]]]:
    """
    Assuming all stations were fetched with same interval and begin_date, their
    timestamps should line up (or be close). We:

    - Take the intersection of timestamps across nodes.
    - Return sorted list of times + per-station arrays of current (knots).
    """
    # Collect sets of times per station
    time_sets = []
    for _, ts in node_series.items():
        time_sets.append({t for (t, _) in ts})

    common_times = sorted(set.intersection(*time_sets))
    currents_by_station: dict[str, list[float]] = {
        sid: [] for sid in node_series.keys()
    }

    for sid, ts in node_series.items():
        ts_dict = {t: v for (t, v) in ts}
        for t in common_times:
            currents_by_station[sid].append(ts_dict[t])

    return common_times, currents_by_station

# --------------------------------------------------------------------
# SPATIAL INTERPOLATION
# --------------------------------------------------------------------

def interpolate_segment_current(
    x_km: float,
    nodes: list[RiverNode],
    node_currents_knots: dict[str, float],
) -> float:
    """
    Interpolate along-river current at position x_km using width-weighted nodal
    currents.

    - River coordinate increases *downstream*.
    - nodes must be sorted by channel_km ascending.
    - node_currents_knots: station_id -> current (knots, downstream-positive).

    Strategy:
      1. Find the segment [x_i, x_{i+1}] containing x.
      2. Compute linear position weight λ in [0,1].
      3. Compute "effective" weights proportional to 1/width at each end
         so narrower node gets a bit more influence.
      4. Linearly blend the currents with those weights.

    If x is beyond the first/last node, extrapolate using nearest segment.
    """
    if x_km <= nodes[0].channel_km:
        i = 0
        lam = 0.0
    elif x_km >= nodes[-1].channel_km:
        i = len(nodes) - 2
        lam = 1.0
    else:
        for j in range(len(nodes) - 1):
            x0 = nodes[j].channel_km
            x1 = nodes[j + 1].channel_km
            if x0 <= x_km <= x1:
                i = j
                lam = (x_km - x0) / (x1 - x0) if x1 != x0 else 0.0
                break
        else:
            # Fallback (shouldn't happen)
            i = len(nodes) - 2
            lam = 1.0

    n0 = nodes[i]
    n1 = nodes[i + 1]
    u0 = node_currents_knots[n0.station_id]
    u1 = node_currents_knots[n1.station_id]

    # Width-based weights: w_eff ~ 1/width
    w0_eff = 1.0 / max(n0.width_m, 1.0)
    w1_eff = 1.0 / max(n1.width_m, 1.0)

    # Blend weights along the segment
    w0 = (1 - lam) * w0_eff
    w1 = lam * w1_eff
    denom = w0 + w1 if (w0 + w1) != 0 else 1.0

    u_interp = (w0 * u0 + w1 * u1) / denom
    return u_interp

# --------------------------------------------------------------------
# TRIP SIMULATION
# --------------------------------------------------------------------

def simulate_trip(
    nodes: list[RiverNode],
    times: list[datetime],
    currents_by_station: dict[str, list[float]],
    start_position_km: float,
    end_position_km: float,
    boat_speed_kts: float = BOAT_SPEED_KTS,
) -> tuple[timedelta, list[tuple[datetime, float, float, float]]]:
    """
    Simple forward-Euler integration of boat position over time.

    Inputs:
      - nodes: ordered from mouth (0) upstream.
      - times: list of datetimes, equally spaced.
      - currents_by_station: station_id -> [current_knots_at_each_time]
      - start_position_km, end_position_km: along-river coordinate.
      - boat_speed_kts: speed through water (constant).

    Returns:
      (trip_duration, history)

      where history is [(time, x_km, current_knots, ground_speed_knots), ...]
      up to arrival or end of available data.

    If we run out of current data before arriving, raises a RuntimeError.
    """
    assert len(times) >= 2, "Need at least 2 time points for integration."

    dt_sec = (times[1] - times[0]).total_seconds()
    if dt_sec <= 0:
        raise ValueError("Non-positive time step in times.")

    # boat_speed_kmh = knots_to_km_per_hour(boat_speed_kts)
    # boat_speed_kmps = km_per_hour_to_km_per_sec(boat_speed_kmh)

    # Determine direction: +1 if going downstream, -1 if going upstream
    direction_sign = 1.0 if end_position_km > start_position_km else -1.0

    x = start_position_km
    history: list[tuple[datetime, float, float, float]] = []

    station_ids = [n.station_id for n in nodes]

    for idx, t in enumerate(times):
        # Build nodal currents at this time
        node_currents = {
            sid: currents_by_station[sid][idx] for sid in station_ids
        }

        # Interpolate current at boat location
        u_knots = interpolate_segment_current(x, nodes, node_currents)

        # Ground speed in knots:
        ground_speed_knots = direction_sign * boat_speed_kts + u_knots
        # Prevent pathological backwards drift with a small floor
        min_forward_kts = 0.1  # you can tune this
        if ground_speed_knots * direction_sign <= 0:
            # Essentially not making progress or going backwards;
            # clamp to a tiny forward speed to avoid infinite loop.
            ground_speed_knots = direction_sign * min_forward_kts

        ground_speed_kmh = knots_to_km_per_hour(ground_speed_knots)
        ground_speed_kmps = km_per_hour_to_km_per_sec(ground_speed_kmh)

        # Update position
        x_new = x + ground_speed_kmps * dt_sec

        history.append((t, x, u_knots, ground_speed_knots))

        # Check if we've crossed the end position
        if (direction_sign > 0 and x_new >= end_position_km) or (
            direction_sign < 0 and x_new <= end_position_km
        ):
            # Linear interpolate final step for more accurate arrival time
            overshoot = x_new - end_position_km
            step_distance = x_new - x
            if step_distance == 0:
                frac = 0.0
            else:
                frac = 1.0 - overshoot / step_distance

            arrival_dt = t + timedelta(seconds=dt_sec * frac)
            trip_duration = arrival_dt - times[0]
            return trip_duration, history

        x = x_new

    raise RuntimeError(
        "Ran out of prediction time window before reaching destination. "
        "Increase `hours_ahead` when fetching currents."
    )

# --------------------------------------------------------------------
# MAIN EXAMPLE
# --------------------------------------------------------------------

def main():
    # Example usage:
    #
    # Trip from Green Cove Springs down to Mouth, starting now (UTC).
    #
    # In practice, you probably want to enter start time in local
    # Eastern time and convert to UTC before calling NOAA.
    start_utc = datetime.now(UTC)

    # Simulation horizon: how many hours of predictions to pull.
    # Must be <= MAX_HOURS_AHEAD.
    hours_ahead = 48

    # Make sure nodes are sorted by channel_km
    nodes = sorted(NODE_CONFIG, key=lambda n: n.channel_km)

    # Fetch up-to-date current predictions for each node
    node_series = build_node_time_series(nodes, start=start_utc, hours=hours_ahead)

    # Reindex onto common time grid
    times, currents_by_station = reindex_to_common_times(node_series)

    # Define start / end positions in km (GCS -> Mouth, or reverse)
    mouth_km = nodes[0].channel_km
    gcs_km = nodes[-1].channel_km

    # Example 1: GCS -> Mouth (downstream)
    start_pos = gcs_km
    end_pos = mouth_km

    trip_duration, history = simulate_trip(
        nodes=nodes,
        times=times,
        currents_by_station=currents_by_station,
        start_position_km=start_pos,
        end_position_km=end_pos,
        boat_speed_kts=BOAT_SPEED_KTS,
    )

    print(f"Estimated trip duration GCS -> Mouth: {trip_duration}")
    print(f"Number of time steps simulated: {len(history)}")
    # You can also dump `history` to CSV if you like.

    # Example 2: Mouth -> GCS (upstream) – demonstrates reversibility
    trip_duration_up, _ = simulate_trip(
        nodes=nodes,
        times=times,
        currents_by_station=currents_by_station,
        start_position_km=mouth_km,
        end_position_km=gcs_km,
        boat_speed_kts=BOAT_SPEED_KTS,
    )
    print(f"Estimated trip duration Mouth -> GCS: {trip_duration_up}")


if __name__ == "__main__":
    main()
