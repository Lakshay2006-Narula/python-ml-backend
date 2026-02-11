import pandas as pd
import folium
from pathlib import Path
import colorsys


def hsv_to_hex(h, s=0.85, v=0.85):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def generate_session_colored_map(df, output_html: str):
    df = df.dropna(subset=["lat", "lon", "session_id"]).copy()
    sessions = sorted(df["session_id"].unique())
    colors = [hsv_to_hex(i / max(1, len(sessions))) for i in range(len(sessions))]

    m = folium.Map(tiles="CartoDB positron", prefer_canvas=True)

    for i, sid in enumerate(sessions):
        seg = df[df["session_id"] == sid].sort_values("timestamp")
        coords = list(zip(seg["lat"], seg["lon"]))
        if not coords:
            continue
        folium.PolyLine(locations=coords, color=colors[i % len(colors)], weight=5, opacity=0.85, tooltip=f"Session {sid}").add_to(m)

    # Add a simple legend as HTML
    legend_html = "<div style='position: absolute; right: 10px; top: 10px; background: white; padding: 8px; border:1px solid #ccc;'>"
    for i, sid in enumerate(sessions[:10]):
        legend_html += f"<div style='margin:2px;'><span style='display:inline-block;width:14px;height:10px;background:{colors[i]};margin-right:6px;'></span>Session {sid}</div>"
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    m.fit_bounds([[df["lat"].min(), df["lon"].min()], [df["lat"].max(), df["lon"].max()]])
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_html)


def main():
    csv = Path("data/processed/filtered_data.csv")
    assert csv.exists(), f"Filtered data not found: {csv}"
    df = pd.read_csv(csv)
    out = "data/tmp/session_colored_routes.html"
    generate_session_colored_map(df, out)
    print("Saved session-colored map to:", out)


if __name__ == "__main__":
    main()
