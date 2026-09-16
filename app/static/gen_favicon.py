"""生成 PNG/ICO favicon —— 与 logo.svg 完全同款的纯 Python 光栅化。

不依赖 Pillow：zlib + struct 手写 PNG 编码，4x 超采样抗锯齿。
产出：favicon.png (256/64/32/16)、apple-touch-icon.png (180)。
"""
import struct
import zlib
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent

# ---------- 几何定义（与 logo.svg 的 64x64 viewBox 一致） ----------
SIZE = 64.0
GRAD_A = (99, 102, 241)    # #6366f1
GRAD_B = (168, 85, 247)    # #a855f7
STAR_B = (147, 51, 234)    # #9333ea
BUBBLE_ALPHA = 0.96

# 气泡圆角矩形：x∈[12,52], y∈[15,41], r=6
BUBBLE = (12.0, 15.0, 52.0, 41.0, 6.0)
# 气泡尾巴（三角近似 SVG 的 l-8.4 7.2 一段）
TAIL = [(29.0, 41.0), (20.6, 48.2), (18.0, 41.5)]
# 四角星芒：8 个顶点（中心 32,26.5）
STAR_C = (32.0, 26.5)
STAR_PTS = [(32, 17.5), (34.6, 23.9), (41, 26.5), (34.6, 29.1),
            (32, 35.5), (29.4, 29.1), (23, 26.5), (29.4, 23.9)]


def in_rounded_rect(px, py, rect):
    x0, y0, x1, y1, r = rect
    if not (x0 <= px <= x1 and y0 <= py <= y1):
        return False
    cx = min(max(px, x0 + r), x1 - r)
    cy = min(max(py, y0 + r), y1 - r)
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def in_polygon(px, py, pts):
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            xin = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < xin:
                inside = not inside
    return inside


def lerp(c1, c2, t):
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def sample(wx, wy):
    """单点颜色（64 坐标系）。返回 (r, g, b)。"""
    # 背景对角渐变
    t = (wx + wy) / (2 * SIZE)
    color = lerp(GRAD_A, GRAD_B, t)
    # 白色气泡（含尾巴）
    if in_rounded_rect(wx, wy, BUBBLE) or in_polygon(wx, wy, TAIL):
        color = lerp(color, (255, 255, 255), BUBBLE_ALPHA)
        # 星芒：渐变紫，叠加在气泡之上
        if in_polygon(wx, wy, [(STAR_C[0] + (p[0] - STAR_C[0]) * 1.0,
                                STAR_C[1] + (p[1] - STAR_C[1]) * 1.0) for p in STAR_PTS]):
            ts = (wx - 23) / 18.0
            color = lerp(GRAD_A, STAR_B, ts)
    return color


def render(size, ss=4):
    """渲染 size x size PNG 像素矩阵（4x 超采样）。"""
    scale = SIZE / size
    rows = []
    for oy in range(size):
        row = bytearray()
        for ox in range(size):
            r = g = b = 0
            for sy in range(ss):
                for sx in range(ss):
                    wx = (ox + (sx + 0.5) / ss) * scale
                    wy = (oy + (sy + 0.5) / ss) * scale
                    c = sample(wx, wy)
                    r += c[0]; g += c[1]; b += c[2]
            n = ss * ss
            row += bytes((round(r / n), round(g / n), round(b / n)))
        rows.append(bytes(row))
    return rows


def write_png(path, rows):
    h = len(rows); w = len(rows[0]) // 3   # rows 每像素 3 字节
    raw = b"".join(b"\x00" + r for r in rows)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    Path(path).write_bytes(png)
    print(f"  {path}  {w}x{h}")


def build_ico(path, png_blobs):
    """ICO 容器（PNG 压缩条目，Vista+ 与全部现代浏览器支持）。"""
    count = len(png_blobs)
    header = struct.pack("<HHH", 0, 1, count)
    offset = 6 + 16 * count
    entries = b""
    body = b""
    for size, data in png_blobs:
        b = 0 if size >= 256 else size   # ICO 规范：256 用 0 表示
        entries += struct.pack("<BBBBHHII", b, b, 0, 0,
                               1, 32, len(data), offset)
        body += data
        offset += len(data)
    Path(path).write_bytes(header + entries + body)
    print(f"  {path}  ICO({', '.join(str(s) for s, _ in png_blobs)})")


if __name__ == "__main__":
    print("光栅化渲染（4x 超采样）…")
    blobs = []
    for size in (16, 32, 64, 256):
        rows = render(size, ss=4 if size <= 64 else 2)
        # write_png 输出同时收集 blob
        h = len(rows); w = len(rows[0]) // 3   # rows 每像素 3 字节
        raw = b"".join(b"\x00" + r for r in rows)

        def chunk(tag, data):
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(raw, 9))
        png += chunk(b"IEND", b"")
        name = "favicon.png" if size == 256 else f"favicon-{size}.png"
        (OUT_DIR / name).write_bytes(png)
        print(f"  {name}  {size}x{size}")
        blobs.append((size if size <= 64 else 256, png))
    build_ico(OUT_DIR / "favicon.ico", blobs)
    rows = render(180, ss=2)
    write_png(OUT_DIR / "apple-touch-icon.png", rows)
    print("完成")
