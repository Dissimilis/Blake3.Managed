"""Generate matching SVG charts, Markdown tables and CSV data from BDN results.

No dependencies. SVGs remain readable in GitHub's light and dark themes.
Usage: python make_chart.py <bdn-log-or-csv> <out.svg> [--xof] [--table out.md]
       [--data out.csv] [--context "machine, runtime, date"]
"""
import argparse
import csv
from html import escape
import math
from pathlib import Path
import re

# Description match, public label, colour. Keep the span overload aligned with the table.
ONESHOT = [
    ("Blake3.Native Rust", "Blake3.Native 3.0.2", "#707070"),
    ("Blake3 3.x xoofx", "Blake3 3.0.2 (xoofx)", "#bc620c"),
    ("CryptoHives package", "CryptoHives 0.6.101", "#8055ad"),
    ("ours Hash(input, output)", "Blake3.Managed (this library)", "#24783c"),
    ("SHA256", "SHA256 (.NET)", "#286fa6"),
]
XOF = [
    ("NativeXof", "Blake3.Native 3.0.2", "#707070"),
    ("CryptoHivesXof", "CryptoHives 0.6.101", "#8055ad"),
    ("AfterXof", "Blake3.Managed (this library)", "#24783c"),
]
UNITS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def duration(value):
    value = value.replace("μ", "u").replace("µ", "u").replace("*", "")
    match = re.fullmatch(r"([\d.,]+)\s*(ns|us|ms|s)", value.strip())
    if not match:
        raise ValueError(f"Invalid BDN duration: {value!r}")
    return float(match[1].replace(",", "")) * UNITS[match[2]]


def parse(path):
    """Read published CSV data or the summary table in an English BDN log."""
    if Path(path).suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as source:
            return [(r["method"], int(r["size_bytes"]), float(r["mean_ns"]),
                     float(r["error_ns"])) for r in csv.DictReader(source)]
    rows = []
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or not cells[1].isdigit():
            continue
        rows.append((cells[0].strip("'"), int(cells[1]), duration(cells[2]), duration(cells[3])))
    return rows


def human(size):
    if size >= 1 << 20 and size % (1 << 20) == 0:
        return f"{size >> 20} MiB"
    if size >= 1 << 10 and size % (1 << 10) == 0:
        return f"{size >> 10} KiB"
    return f"{size} B"


def select(rows, series):
    data = {}
    for key, label, colour in series:
        points = sorted((size, mean, error) for desc, size, mean, error in rows if key in desc)
        if not points or len({s for s, _, _ in points}) != len(points):
            raise ValueError(f"Missing or duplicate measurements for {label}")
        if any(mean <= 0 or error < 0 or not math.isfinite(mean + error) for _, mean, error in points):
            raise ValueError(f"Invalid measurements for {label}")
        data[label] = (points, colour)
    sizes = [s for s, _, _ in next(iter(data.values()))[0]]
    if len(sizes) < 2 or sizes[0] <= 0:
        raise ValueError("A chart needs at least two positive sizes")
    if any([s for s, _, _ in points] != sizes for points, _ in data.values()):
        raise ValueError("All series must cover the same sizes")
    return data, sizes


def table(data, sizes, xof):
    lines = ["| " + " | ".join(["Output" if xof else "Input", *data]) + " |",
             "|" + "---:|" * (len(data) + 1)]
    for index, size in enumerate(sizes):
        native_mean = next(iter(data.values()))[0][index][1]
        unit = "ms" if native_mean >= 1e6 else "us" if native_mean >= 1e3 else "ns"
        cells = [human(size)]
        for points, _ in data.values():
            _, mean, error = points[index]
            cell = f"{mean / UNITS[unit]:.2f} ± {error / UNITS[unit]:.2f} {unit}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def chart(data, sizes, xof, context):
    width, height = 1200, 630
    left, right, top, bottom = 80, 305, 80, 80
    pw, ph = width - left - right, height - top - bottom
    values = [mean / 1000 if xof else size / mean
              for points, _ in data.values() for size, mean, _ in points]
    y_lo, y_hi = math.log10(min(values) * .72), math.log10(max(values) * 1.35)
    x_lo, x_hi = math.log2(sizes[0]), math.log2(sizes[-1])
    def x(size):
        return left + (math.log2(size) - x_lo) / (x_hi - x_lo) * pw
    def y(value):
        return top + ph - (math.log10(value) - y_lo) / (y_hi - y_lo) * ph
    title = "BLAKE3 XOF latency — lower is better" if xof else "BLAKE3 throughput — higher is better"
    detail = ("Two 1 KiB absorbs, output, reset. All implementations use one thread."
              if xof else "One-shot hashing into a 32-byte destination. Shading marks the parallel range.")
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc" '
           'font-family="-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">',
           f'<title id="title">{escape(title)}</title>',
           f'<desc id="desc">{escape(detail)} Both axes use logarithmic scales. '
           'Points show measured means; the accompanying table gives confidence margins.</desc>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{left}" y="28" font-size="20" font-weight="600" fill="#202020">{escape(title)}</text>',
           f'<text x="{left}" y="49" font-size="12" fill="#555">{escape(context)}</text>',
           f'<text x="{left}" y="67" font-size="11" fill="#555">{escape(detail)}</text>']
    if not xof and sizes[0] < 72 * 1024 < sizes[-1]:
        threshold = x(72 * 1024)
        out.extend([f'<rect x="{threshold:.1f}" y="{top}" width="{left+pw-threshold:.1f}" height="{ph}" fill="#f0f7f1"/>',
                    f'<line x1="{threshold:.1f}" y1="{top}" x2="{threshold:.1f}" y2="{top+ph}" stroke="#9dbca4" stroke-dasharray="4 4"/>'])
    for decade in range(math.floor(y_lo), math.ceil(y_hi) + 1):
        for mantissa in (1, 2, 5):
            tick = mantissa * 10 ** decade
            if y_lo <= math.log10(tick) <= y_hi:
                yy = y(tick)
                out.extend([f'<line x1="{left}" y1="{yy:.1f}" x2="{left+pw}" y2="{yy:.1f}" stroke="#e4e7e5"/>',
                            f'<text x="{left-10}" y="{yy+4:.1f}" text-anchor="end" font-size="11" fill="#555">{tick:g}</text>'])
    # Omit 6 KiB's axis label between 4 and 8 KiB; its measurement remains plotted and in the table.
    ticks = [size for size in sizes if size != 6144 or 4096 not in sizes or 8192 not in sizes]
    for size in ticks:
        xx = x(size)
        out.extend([f'<line x1="{xx:.1f}" y1="{top+ph}" x2="{xx:.1f}" y2="{top+ph+5}" stroke="#999"/>',
                    f'<text x="{xx:.1f}" y="{top+ph+21}" text-anchor="middle" font-size="10.5" fill="#555">{human(size)}</text>'])
    axis = "Time per absorb / output / reset (us, log scale)" if xof else "Throughput (GB/s, log scale)"
    out.extend([f'<text x="20" y="{top+ph/2}" transform="rotate(-90 20 {top+ph/2})" text-anchor="middle" font-size="12" fill="#444">{axis}</text>',
                f'<line x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}" stroke="#999"/>',
                f'<text x="{left+pw/2}" y="{height-28}" text-anchor="middle" font-size="12" fill="#444">{"Output" if xof else "Input"} size (log scale)</text>'])
    ly = top + 16
    for label, (points, colour) in data.items():
        path = " ".join(("M" if i == 0 else "L") + f"{x(size):.1f},{y(mean/1000 if xof else size/mean):.1f}"
                        for i, (size, mean, _) in enumerate(points))
        weight = 3 if "this library" in label else 2
        out.append(f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="{weight}" stroke-linejoin="round"/>')
        for size, mean, error in points:
            value = mean / 1000 if xof else size / mean
            tooltip = f"{label}, {human(size)}: {mean / 1000:.4g} ± {error / 1000:.3g} us"
            out.append(f'<circle cx="{x(size):.1f}" cy="{y(value):.1f}" r="3.5" fill="{colour}"><title>{escape(tooltip)}</title></circle>')
        out.extend([f'<line x1="{left+pw+25}" y1="{ly-4}" x2="{left+pw+49}" y2="{ly-4}" stroke="{colour}" stroke-width="{weight}"/>',
                    f'<text x="{left+pw+59}" y="{ly}" font-size="12" fill="#333">{escape(label)}</text>'])
        ly += 30
    notes = (["All implementations: one thread.", "Same 1 KiB input absorbed twice."] if xof else
             ["This library: multiple cores", "above 72 KiB; one thread below.", "Other implementations: one thread."])
    notes += ["", "Means shown; see table for margins.", "Lines connect measured sizes.", "Compare within this run only."]
    for note in notes:
        out.append(f'<text x="{left+pw+25}" y="{ly+25}" font-size="11" fill="#666">{escape(note)}</text>')
        ly += 18
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--xof", action="store_true")
    parser.add_argument("--table")
    parser.add_argument("--data")
    parser.add_argument("--context", default="See README for hardware, runtime and measurement details.")
    args = parser.parse_args()
    rows = parse(args.input)
    data, sizes = select(rows, XOF if args.xof else ONESHOT)
    Path(args.output).write_text(chart(data, sizes, args.xof, args.context), encoding="utf-8")
    if args.table:
        Path(args.table).write_text(table(data, sizes, args.xof), encoding="utf-8")
    if args.data:
        selected = {key for key, _, _ in (XOF if args.xof else ONESHOT)}
        with open(args.data, "w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["method", "size_bytes", "mean_ns", "error_ns"])
            writer.writerows(row for row in rows if any(key in row[0] for key in selected))
    print(f"Wrote {args.output}: {len(data)} series, {len(sizes)} sizes")


if __name__ == "__main__":
    main()
