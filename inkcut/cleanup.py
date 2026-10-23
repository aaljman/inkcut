"""Remove the leftover paper colour from a cut-out of art on a flat background.

clean(src, a0) -> (rgb, alpha, info)

The input is the original image and the alpha mask a segmentation model made for
it. When the background is one flat colour B (white or cream paper, a flat grey,
a photo studio backdrop), every edge pixel is a mix of the subject and B, and B
can be solved out of it exactly. Four stages:

1. Edges: each pixel in the model's soft edge band is split into the colour of
   the stroke it belongs to plus paper; alpha is recomputed and the paper colour
   removed. Only pixels outside the outline (reachable from the background
   without crossing ink) may lose alpha; inside the outline alpha can only rise.
2. Slivers: 1-2 px gaps of paper between ink lines that the model marked solid.
3. Paper caught between ink and a coloured fill: split into ink + fill + paper,
   the paper share removed when that split explains the pixel closely.
4. Paint that spilled past the outline onto the paper: only the ink share stays.

Protected throughout: wide light areas (eye whites and their shading, white cloth,
teeth), solid paint, and light paint that is whiter than the paper (a sticker's
white border on cream paper).
"""
import numpy as np
from scipy import ndimage

P = {
    "BAND_PX": 3,        # grow the model's soft edge by this much
    "MIN_SPREAD": 60.0,  # |F - B| below this: subject too close to the paper colour, keep model alpha
    "T_LIGHT": 0.15,     # how far toward paper a pixel must be to count as light
    "PERP": 35.0,        # max distance off the ink -> paper colour line (rejects coloured paint)
    "PERP_MIX": 18.0,    # stage 2: only pixels this close to pure ink + paper get new alpha
    "T_INK": 0.15,       # darker than this on the ink -> paper line counts as ink
    "PERP_INK": 20.0,    # stage 2: sliver-bounding ink must also be this close to neutral
    "PAPER_MATCH": 12.0, # a patch whose lightest pixel is this far from the paper colour...
    "T_OVER": 0.01,      # ...and past the paper on the ink -> paper line (whiter than paper) is paint
    "LP_R": 2,           # light paint must contain a disk this wide (texture specks do not)
    "LP_A0": 0.2,        # light paint only where the model gave at least this alpha
    "EDGE_EXT": 10,      # stage 1: re-solve only within this many px of the background, outside the outline
    "T_INK_CORE": 0.25,  # stage 1: a stroke this dark on the ink -> paper line is ink: its halo may be cut
    "PAPER_PURE": 10.0,  # stage 1: pixels this close to the paper colour may always be cut
    "OUTLINE_MIN": 0.5,  # share of the silhouette traced by thin ink above which an image is line art
    "OUTLINE_THICK": 4,  # dark areas a disk this wide fits in are dark objects, not ink lines
    "PHOTO_SIGMA": 5.0,  # photo mode: blur radius for the colour of low-alpha edge pixels
    "PHOTO_T0": 0.5,     # photo mode: below this alpha the blurred colour takes over...
    "PHOTO_TW": 0.4,     # ...fully, blending to the exact un-mix over this alpha range
    "TEX_HF": 12.0,      # printed texture: pixels differing this much from their blurred surroundings...
    "TEX_WIN": 25,       # ...not next to ink, counted over a window this wide...
    "TEX_MAX": 0.12,     # ...covering more than this share of it (hatching, line screen, halftone)
    "GAP_LIGHT": 90.0,   # ink hatching: paper-coloured pixels this close to the paper colour...
    "GAP_WIN": 41,       # ...in thin gaps, counted over a window this wide...
    "GAP_MAX": 0.05,     # ...covering more than this share of it (line art scores ~0-0.03)
    "TEX_OPEN": 12,      # printed texture must fill a disk this wide (drops specks along sketchy lines)
    "TEX_GROW": 6,       # grow printed-texture areas by this much (covers their thin outer tips)
    "T_BARRIER": 0.4,    # stage 1: ink up to this far toward paper blocks the outside flood
    "SLIVER_REACH": 10,  # stage 2: a sliver must come within this many px of the background...
    "HAIRLINE_R": 2.5,   # ...or be a hairline (inscribed radius at most this, incl. anti-aliasing)...
    "HAIRLINE_PX": 40,   # ...at least this many px long
    "COS_BAD": 0.5,      # stage 1: chroma cosine below this vs the nearest solid = a different stroke
    "CHROMA_MIN": 25.0,  # stage 1: ...judged only when both colours carry at least this much chroma
    "COS_MIN": 0.92,     # stage 1: a stroke core must point the same way from the paper as the pixel
    "EDGE_R": 6,         # stage 1: search radius for that stroke core
    "CORE_MIN": 120.0,   # stage 1: a core must be at least this far from the paper colour
    "PAPER_DIST": 30.0,  # stage 1: pixels this close to the paper colour are paper, never searched
    "R_FAT": 3.0,        # light areas a disk of this radius fits in are real white features
    "PROTECT_PX": 3,     # also protect this far along a wide feature's tapering ends
    "INK_RING": 0.3,     # stage 2: share of a patch's 2-px ring that must be ink (or background)
    "GROW": 2,           # stage 2: also re-solve this many px of anti-aliasing round a sliver
    "STAGE2": 1,
    "STAGE3": 1,
    "STAGE4": 1,
    "FRINGE_PX": 4,      # stage 4: how far from the background the outside-the-outline zone reaches
    "FILL_GAP": 1,       # stage 4: skip pixels this close to solid coloured paint (outline-less edges)
    "UNMIX_MAX": 90.0,   # un-mixed colours this far from both ink and nearby paint are noise
    "FILL_DIST": 60.0,   # solid pixels at least this far from the paper colour count as fill
    "T_PAINT": 0.06,     # grey paint: lighter than this on the ink -> paper line (and wide) is paint
    "RES": 14.0,         # stage 3: max error of the ink + fill + paper mix to trust it
    "PAPER_MIN": 0.15,   # stages 3-4: only pixels holding at least this much paper change
    "NEAR_PX": 2,        # stage 3: only within this many px of ink or the background
    "WIDE_DIST": 110.0,  # wide light features: pixels this close to the paper colour
    "T_SHADE": 0.4,      # wide light features: their shading, this far toward paper, stays with them
    "FLAT_MAX": 30.0,    # background counts as flat when 95% of it is within this of its median
                         # (or, for a shadow / vignette / aged paper, within this of its smoothed colour)
    "BG_SIGMA_DIV": 150.0,  # smoothing radius for a varying background: image short side / this
}
DEFAULT_INK = np.array([25.0, 25.0, 25.0], np.float32)


def disk(r):
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r + r


def line_pos(img, K, B):
    """Position on the ink -> paper colour line (t: 0 ink, 1 paper) and distance off it (perp)."""
    d = np.broadcast_to(B - K, img.shape)
    v = img - K
    t = np.einsum("ijk,ijk->ij", v, d) / np.maximum(np.einsum("ijk,ijk->ij", d, d), 1e-6)
    return t, np.linalg.norm(v - t[..., None] * d, axis=2)


def unmix(src, a, B, fallback, K=None):
    """Remove the paper colour B; faint pixels lean on fallback instead of amplifying noise.

    With K given, an un-mixed colour far from both the ink and the fallback colour
    (a hue the image never had) snaps to the nearer of the two.
    """
    am = a[..., None]
    unmixed = np.clip((src - (1 - am) * B) / np.maximum(am, 1e-6), 0, 255)
    if K is not None:
        d_fb = np.linalg.norm(unmixed - fallback, axis=2)
        d_k = np.linalg.norm(unmixed - K, axis=2)
        stray = (d_fb > P["UNMIX_MAX"]) & (d_k > P["UNMIX_MAX"])
        nearer = np.where((d_k < d_fb)[..., None], np.broadcast_to(K, unmixed.shape), fallback)
        unmixed = np.where(stray[..., None], nearer, unmixed)
    trust = np.clip((a - 0.2) / 0.4, 0, 1)[..., None]
    return trust * unmixed + (1 - trust) * fallback


def solid_fill(src, a, B, K):
    """Solid paint: coloured pixels, plus wide neutral paint (grey metal, grey cloth).

    Thin neutral pixels between ink and paper are anti-aliasing, not paint.
    """
    t, perp = line_pos(src, K, B)
    solid = (a > 0.98) & (np.linalg.norm(src - B, axis=2) > P["FILL_DIST"])
    # Dark grey shading counts when it is wide; ink-line edges are only 1-2 px of grey.
    neutral_paint = solid & (perp < P["PERP"]) & (t > P["T_PAINT"])
    neutral_paint = ndimage.binary_opening(neutral_paint, structure=disk(2))
    return solid & ((perp >= P["PERP"]) | neutral_paint)


def wide_light_features(src, a0, B, K):
    """Light areas a disk fits in (eye whites, white cloth), grown over their shading and rims."""
    t, perp = line_pos(src, K, B)
    light = (a0 > 0.5) & (t > P["T_LIGHT"]) & (np.linalg.norm(src - B, axis=2) < P["WIDE_DIST"])
    core = ndimage.binary_opening(light, structure=disk(int(P["R_FAT"])))
    loose = (a0 > 0.5) & (t > P["T_SHADE"]) & (perp < P["PERP"])
    rim = ndimage.binary_propagation(core, mask=loose | core)
    return ndimage.binary_dilation(rim, structure=np.ones((3, 3)), iterations=P["PROTECT_PX"])


def light_paint(src, B, K):
    """Whiter than the paper (past it on the ink -> paper line, clearly off its colour) and part of a
    wide region, like a sticker's white border on cream paper. Paper texture specks are not."""
    t, _ = line_pos(src, K, B)
    raw = (t > 1.0 + P["T_OVER"]) & (np.linalg.norm(src - B, axis=2) > P["PAPER_MATCH"])
    core = ndimage.binary_opening(raw, structure=disk(P["LP_R"]))
    return ndimage.binary_propagation(core, mask=raw)


def printed_texture(src, a0, B):
    """Areas busy with fine detail away from ink lines: hatching, line screens, halftone dots,
    stipple. The paper there is part of the printed colour, not a gap to cut out. Detail right
    next to ink lines is just outlines and is not counted, so line art scores ~0-0.08 while a
    printed line screen scores ~0.2."""
    lum = src.mean(axis=2)
    active = np.abs(lum - ndimage.gaussian_filter(lum, 1.5)) > P["TEX_HF"]
    ink = (lum < 90) & (src.max(axis=2) - src.min(axis=2) < 50)
    active &= ~ndimage.binary_dilation(ink, iterations=2)
    density = ndimage.uniform_filter(active.astype(np.float32), P["TEX_WIN"])
    # Ink hatching (fur, pen shading) is ink lines with paper between them: dense thin paper gaps.
    light = (np.linalg.norm(src - B, axis=2) < P["GAP_LIGHT"]) & (a0 > 0.05)
    gaps = light & ~ndimage.binary_opening(light, structure=disk(2))
    gap_density = ndimage.uniform_filter(gaps.astype(np.float32), P["GAP_WIN"])
    busy = (density > P["TEX_MAX"]) | (gap_density > P["GAP_MAX"])
    # Printed texture covers areas; scattered busy specks along sketchy outlines do not count.
    texture = ndimage.binary_opening(busy & (a0 > 0.05), structure=disk(P["TEX_OPEN"]))
    return ndimage.binary_dilation(texture, iterations=P["TEX_GROW"])


def outline_coverage(src, a0):
    """Share of the subject's silhouette traced by a dark ink line: ~1 for outlined line art,
    ~0 for photos, paintings, engravings and stickers."""
    fg = a0 > 0.5
    edge = fg & ~ndimage.binary_erosion(fg, iterations=2)
    lum, chroma = src.mean(axis=2), src.max(axis=2) - src.min(axis=2)
    ink = (lum < 70) & (chroma < 40)
    # An ink outline is a thin line; a dark animal or object is dark all the way in.
    thick = ndimage.binary_opening(ink, structure=disk(P["OUTLINE_THICK"]))
    thin = ink & ~ndimage.binary_dilation(thick, iterations=1)
    near_ink = ndimage.binary_dilation(thin, iterations=3)
    return (edge & near_ink).sum() / max(edge.sum(), 1)


def stage_edges(src, a0, B, K, lineart=True, texture=None):
    solid = a0 > 0.98
    soft = (a0 > 0.01) & ~solid
    band = ndimage.binary_dilation(soft, iterations=P["BAND_PX"]) & ~ndimage.binary_erosion(solid, iterations=1)
    reach = ndimage.binary_dilation(a0 > 0.02, iterations=P["BAND_PX"])
    band &= reach

    _, (iy, ix) = ndimage.distance_transform_edt(~solid, return_indices=True)
    F = src[iy, ix].copy()

    # The nearest solid colour is usually the stroke an edge pixel belongs to. When its hue clearly
    # disagrees (a detached orange stroke beside a navy area), use the stroke's own core instead:
    # the most saturated same-hue pixel within EDGE_R; a strongly coloured pixel with none is its own
    # core. Hue is compared with brightness taken out, since from light paper most colours point
    # "darker" and look alike.
    CB = src - B
    cc = src - src.mean(axis=2, keepdims=True)
    fc = F - F.mean(axis=2, keepdims=True)
    ncc, nfc = np.linalg.norm(cc, axis=2), np.linalg.norm(fc, axis=2)
    hue_cos = np.einsum("ijk,ijk->ij", cc, fc) / np.maximum(ncc * nfc, 1e-6)
    other_hue = (ncc >= P["CHROMA_MIN"]) & (nfc >= P["CHROMA_MIN"]) & (hue_cos < P["COS_BAD"])
    ys, xs = np.nonzero(band & ~solid & other_hue & (np.linalg.norm(CB, axis=2) >= P["PAPER_DIST"]))
    if len(ys):
        cb = CB[ys, xs]
        ncb = np.linalg.norm(cb, axis=1)
        best = ncb.copy()
        f_core = src[ys, xs].copy()
        found = np.zeros(len(ys), bool)
        h, w = a0.shape
        r = P["EDGE_R"]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dy * dy + dx * dx > r * r or (dy == 0 and dx == 0):
                    continue
                yy, xx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
                cand = CB[yy, xx]
                nc = np.linalg.norm(cand, axis=1)
                cos = np.einsum("ij,ij->i", cand, cb) / np.maximum(nc * ncb, 1e-6)
                ok = (cos >= P["COS_MIN"]) & (nc > best) & (nc >= P["CORE_MIN"]) & (a0[yy, xx] > 0.5)
                best[ok] = nc[ok]
                f_core[ok] = src[yy, xx][ok]
                found |= ok
        own = found | (ncb >= P["CORE_MIN"])
        F[ys[own], xs[own]] = f_core[own]

    # Light paint takes its colour from the nearest solid light paint, or is its own colour; it is
    # too close to the paper for MIN_SPREAD, so it gets its own smaller limit.
    lp = light_paint(src, B, K) & band & (a0 >= P["LP_A0"])
    if lp.any():
        lp_solid = light_paint(src, B, K) & (a0 > 0.9)
        if lp_solid.any():
            d_lp, (ly, lx) = ndimage.distance_transform_edt(~lp_solid, return_indices=True)
            near = lp & (d_lp <= P["EDGE_R"])
            F[near] = src[ly[near], lx[near]]
            F[lp & ~near] = src[lp & ~near]
        else:
            F[lp] = src[lp]

    FB = F - B
    spread = np.linalg.norm(FB, axis=2)
    a_col = np.clip(np.einsum("ijk,ijk->ij", src - B, FB) / np.maximum(spread ** 2, 1e-6), 0, 1)

    # Paper only reaches pixels outside the outline. Flood from the background (within EDGE_EXT)
    # without crossing ink: those pixels are re-solved freely. Pixels behind an ink line are inside
    # the subject (teeth, a white frill at the edge): they may gain alpha, never lose it.
    t_s, perp_s = line_pos(src, K, B)
    barrier = (perp_s < P["PERP"]) & (t_s < P["T_BARRIER"])  # ink and its grey anti-aliasing
    ext = a0 < 0.05
    within = ndimage.distance_transform_edt(~ext) <= P["EDGE_EXT"]
    outside_line = ndimage.binary_propagation(ext, structure=np.ones((3, 3)), mask=(within & ~barrier) | ext)
    # The outline's own outer pixels hold paper too.
    outside_line |= barrier & ndimage.binary_dilation(outside_line, structure=np.ones((3, 3)))

    # Alpha is re-solved from colour only where the stroke is ink (a paper halo next to an ink
    # line) or the pixel is plain paper. At the edge of coloured paint or photographed material a
    # pale pixel may simply be pale (a light leaf, a fine white hair): keep the model's alpha
    # there and only take the paper colour out.
    if not lineart:
        # Photos, paintings, engravings: alpha stays the model's (white-on-white detail like a fine
        # hair, or pale paint lighter than the paper, cannot be told from paper by colour); only
        # the paper colour is taken out of the soft edge.
        a = a0.copy()
        a[a < 0.05] = 0
        edge = band & (a > 0)
        am = a[..., None]
        exact = np.clip((src - (1 - am) * B) / np.maximum(am, 1e-6), 0, 255)
        # The exact un-mix magnifies any error in the model's alpha where alpha is low, so there the
        # colour comes from an alpha-weighted blur of the confident neighbours' un-mixed colours.
        w = (a ** 2)[..., None]
        sig = P["PHOTO_SIGMA"]
        num = np.stack([ndimage.gaussian_filter(exact[..., c] * w[..., 0], sig) for c in range(3)], axis=-1)
        smooth = num / np.maximum(ndimage.gaussian_filter(w[..., 0], sig), 1e-6)[..., None]
        trust = np.clip((a - P["PHOTO_T0"]) / P["PHOTO_TW"], 0, 1)[..., None]
        out = src.copy()
        out[edge] = (trust * exact + (1 - trust) * smooth)[edge]
        return a, out, int(edge.sum())

    t_f, perp_f = line_pos(F, K, B)
    ink_core = (perp_f < P["PERP"]) & (t_f < P["T_INK_CORE"])
    paper_px = np.linalg.norm(src - B, axis=2) < P["PAPER_PURE"]
    a = a0.copy()
    use = band & ((spread >= P["MIN_SPREAD"]) | (lp & (spread >= P["PAPER_MATCH"])))
    free = use & outside_line & (ink_core | paper_px | lp)
    if texture is not None:
        free &= ~texture  # paper inside hatching or a line screen is part of the printed colour
    a[free] = a_col[free]
    inner = use & ~outside_line
    a[inner] = np.maximum(a0[inner], a_col[inner])
    a[~reach] = 0
    a[a < 0.05] = 0

    out = src.copy()
    edge = band & (a > 0) & outside_line
    out[edge] = unmix(src, a, B, F, K)[edge]
    return a, out, int(edge.sum())


def stage_slivers(src, a, out, B, K, protect_src):
    t, perp = line_pos(src, K, B)          # source mix: drives the new alpha
    t_now, perp_now = line_pos(out, K, B)  # what stage 1 left: decides what still looks light
    # Slivers must be bound by real ink (near-neutral dark); dark coloured paint is not ink.
    ink = (perp < P["PERP_INK"]) & (t < P["T_INK"])
    outside = a < 0.05

    paint = solid_fill(src, a, B, K)
    light = (perp_now < P["PERP"]) & (t_now > P["T_LIGHT"]) & (a > 0.05) & ~paint
    core = ndimage.binary_opening(light, structure=disk(int(P["R_FAT"])))
    protect = ndimage.binary_dilation(core, structure=np.ones((3, 3)), iterations=P["PROTECT_PX"]) & light
    thin = light & ~protect

    labels, n = ndimage.label(thin, structure=np.ones((3, 3)))
    ring = ndimage.binary_dilation(thin, structure=np.ones((3, 3)), iterations=2) & ~thin
    ring_lab = ndimage.grey_dilation(labels, footprint=disk(2)) * ring
    idx = np.arange(1, n + 1)
    ring_n = ndimage.sum(np.ones_like(a), ring_lab, index=idx)
    ring_ok = ndimage.sum((ink | outside).astype(np.float32), ring_lab, index=idx)
    share = ring_ok / np.maximum(ring_n, 1)
    # A patch whose lightest pixel is whiter than the paper is paint: no ink + paper mix is
    # lighter than the paper. Inert on pure white paper.
    dist_b = np.linalg.norm(src - B, axis=2)
    if n:
        pos = ndimage.maximum_position(t, labels, index=idx)
        py_, px_ = np.array([p[0] for p in pos]), np.array([p[1] for p in pos])
        paper_ok = ~((t[py_, px_] > 1.0 + P["T_OVER"]) & (dist_b[py_, px_] > P["PAPER_MATCH"]))
    else:
        paper_ok = np.zeros(0, bool)
    # Paper shows through near the background (gaps between limbs, loops of hair): within
    # SLIVER_REACH of it, joined to it through light pixels, or a long hairline (a closed gap
    # the model filled). Short white patches deep inside the subject (teeth) are paint.
    d_ext = ndimage.distance_transform_edt(~outside)
    joined = ndimage.binary_propagation(outside, structure=np.ones((3, 3)), mask=light | outside)
    if n:
        width = ndimage.maximum(ndimage.distance_transform_edt(thin), labels, index=idx)
        area = ndimage.sum(np.ones_like(a), labels, index=idx)
        near_ext = (ndimage.minimum(d_ext, labels, index=idx) <= P["SLIVER_REACH"]) | \
                   (ndimage.maximum(joined.astype(np.uint8), labels, index=idx) > 0) | \
                   ((width <= P["HAIRLINE_R"]) & (area >= P["HAIRLINE_PX"]))
    else:
        near_ext = np.zeros(0, bool)
    sliver = np.isin(labels, idx[(share >= P["INK_RING"]) & paper_ok & near_ext])

    zone = sliver
    if P["GROW"]:
        zone = ndimage.binary_dilation(sliver, structure=np.ones((3, 3)), iterations=P["GROW"]) & (a > 0.05) & ~protect
    # Only true ink + paper mixes; ink blended with coloured paint, paint itself and wide light
    # features stay as they are.
    zone &= (perp < P["PERP_MIX"]) & ~paint & ~protect_src

    a2 = a.copy()
    a2[zone] = np.minimum(a[zone], np.clip(1 - t[zone], 0, 1))
    a2[a2 < 0.05] = 0
    out2 = out.copy()
    Kimg = np.broadcast_to(K, src.shape)
    out2[zone] = unmix(src, a2, B, Kimg, Kimg)[zone]
    return a2, out2, int(zone.sum())


def three_way_solve(src, a, B, K):
    """Split each pixel into ink K, nearest fill F and paper B: C = w_ink*K + w_fill*F + w_paper*B."""
    fill = solid_fill(src, a, B, K)
    _, (iy, ix) = ndimage.distance_transform_edt(~fill, return_indices=True)
    F = src[iy, ix]
    u = np.broadcast_to(K - B, src.shape)
    w = F - B
    c = src - B
    uu = np.einsum("ijk,ijk->ij", u, u)
    ww = np.einsum("ijk,ijk->ij", w, w)
    uw = np.einsum("ijk,ijk->ij", u, w)
    uc = np.einsum("ijk,ijk->ij", u, c)
    wc = np.einsum("ijk,ijk->ij", w, c)
    det = uu * ww - uw * uw
    w_ink = np.clip((uc * ww - wc * uw) / np.maximum(det, 1e-6), 0, 1)
    w_fill = np.clip((wc * uu - uc * uw) / np.maximum(det, 1e-6), 0, 1)
    total = w_ink + w_fill
    over = total > 1
    w_ink[over] /= total[over]
    w_fill[over] /= total[over]
    total = np.minimum(total, 1)
    resid = np.linalg.norm(c - w_ink[..., None] * u - w_fill[..., None] * w, axis=2)
    return w_ink, w_fill, total, resid, F, (iy, ix)


def stage_three_way(src, a, out, B, K, protect_src):
    t, perp = line_pos(src, K, B)
    ink = (perp < P["PERP"]) & (t < P["T_INK"])
    outside = a < 0.05
    lum = src.mean(axis=2)
    w_ink, w_fill, total, resid, F, (iy, ix) = three_way_solve(src, a, B, K)

    near = ndimage.binary_dilation(ink | outside, structure=np.ones((3, 3)), iterations=P["NEAR_PX"])
    # Plain ink + paper pixels are stage 2's; this stage only takes tinted mixes.
    cand = (a > 0.05) & near & (perp >= P["PERP_MIX"]) & (resid < P["RES"]) \
        & (1 - total > P["PAPER_MIN"]) & (lum > lum[iy, ix])
    core = ndimage.binary_opening(cand, structure=disk(int(P["R_FAT"])))
    protect = ndimage.binary_dilation(core, structure=np.ones((3, 3)), iterations=P["PROTECT_PX"])
    cand &= ~protect & ~protect_src

    a3 = a.copy()
    a3[cand] = np.minimum(a[cand], total[cand])
    a3[a3 < 0.05] = 0
    out3 = out.copy()
    mixed = (w_ink[..., None] * K + w_fill[..., None] * F) / np.maximum(total[..., None], 1e-6)
    out3[cand] = mixed[cand]
    return a3, out3, int(cand.sum())


def stage_outside_fringe(src, a, out, B, K, protect_src):
    """Outside = reachable from the background within FRINGE_PX without crossing ink."""
    t, perp = line_pos(src, K, B)
    ink = (perp < P["PERP"]) & (t < P["T_INK"])
    outside = a < 0.05
    w_ink, _, total, _, _, _ = three_way_solve(src, a, B, K)
    within = ndimage.distance_transform_edt(~outside) <= P["FRINGE_PX"]
    reach = ndimage.binary_propagation(outside, structure=np.ones((3, 3)), mask=(within & ~ink) | outside)
    # Coloured paint touching the pixel means a normal outline-less edge, not spill.
    paint = (a > 0.5) & (perp >= P["PERP"]) & (np.linalg.norm(src - B, axis=2) > P["FILL_DIST"])
    near_paint = ndimage.binary_dilation(paint, structure=np.ones((3, 3)), iterations=P["FILL_GAP"])
    fringe = reach & ~outside & ~near_paint & (1 - total > P["PAPER_MIN"]) & ~protect_src

    a4 = a.copy()
    a4[fringe] = np.minimum(a[fringe], w_ink[fringe])
    a4[a4 < 0.05] = 0
    out4 = out.copy()
    out4[fringe] = K
    return a4, out4, int(fringe.sum())


def local_background(src, a0):
    """Background colour per pixel, spread smoothly from the visible background (a soft shadow,
    a vignette, aged paper). Near an edge it is the colour of the background right next to it."""
    bgm = (a0 < 0.02).astype(np.float32)
    sigma = max(6.0, min(a0.shape) / P["BG_SIGMA_DIV"])
    den = ndimage.gaussian_filter(bgm, sigma)
    num = np.stack([ndimage.gaussian_filter(src[..., c] * bgm, sigma) for c in range(3)], axis=-1)
    field = num / np.maximum(den, 1e-6)[..., None]
    # Far inside the subject no background is in reach; take the nearest estimated colour.
    known = den > 0.02
    _, (iy, ix) = ndimage.distance_transform_edt(~known, return_indices=True)
    return field[iy, ix]


def clean(src, a0):
    """src: HxWx3 float 0..255, a0: model alpha HxW 0..1.

    Returns (rgb, alpha, info). rgb is None when the cleanup was skipped (background neither flat
    nor smoothly varying).
    """
    bgsel = a0 < 0.02
    bg = src[bgsel]
    if len(bg) < 0.01 * a0.size:
        return None, a0, "skipped: almost no background visible"
    B = np.median(bg, axis=0)
    spread = np.percentile(np.linalg.norm(bg - B, axis=1), 95)
    mode = "flat"
    if spread > P["FLAT_MAX"]:
        # Not one colour: accept a background that varies smoothly (shadow, vignette, aged paper).
        B_field = local_background(src, a0)
        resid = np.percentile(np.linalg.norm(bg - B_field[bgsel], axis=1), 95)
        if resid > P["FLAT_MAX"]:
            return None, a0, (f"skipped: background is neither flat nor smooth "
                              f"(95% spread {spread:.0f}, after smoothing {resid:.0f} > {P['FLAT_MAX']:.0f})")
        B, mode = B_field, f"smooth (spread {spread:.0f} -> {resid:.0f})"
    lum, chroma = src.mean(axis=2), src.max(axis=2) - src.min(axis=2)
    inkpx = src[(a0 > 0.98) & (lum < 60) & (chroma < 25)]
    K = np.median(inkpx, axis=0) if len(inkpx) > 50 else DEFAULT_INK

    # Ink-outlined line art gets the full cleanup. In photos, paintings and engravings a pale
    # pixel can be pale material (a light leaf, a fine white hair), which colour alone cannot
    # tell from paper: there only the paper colour is taken out of the edges and the model's
    # alpha is kept.
    cover = outline_coverage(src, a0)
    lineart = cover >= P["OUTLINE_MIN"]
    texture = printed_texture(src, a0, B) if lineart else None
    a, out, n1 = stage_edges(src, a0, B, K, lineart, texture)
    n2 = n3 = n4 = 0
    if lineart:
        # Light paint is never paper: stages 2-4 leave it (and its 1 px rim) alone.
        lp = ndimage.binary_dilation(light_paint(src, B, K) & (a0 > 0.05), structure=np.ones((3, 3)))
        protect_src = wide_light_features(src, a0, B, K) | lp | texture
        if P["STAGE2"]:
            a, out, n2 = stage_slivers(src, a, out, B, K, protect_src)
        if P["STAGE3"]:
            a, out, n3 = stage_three_way(src, a, out, B, K, protect_src)
        if P["STAGE4"]:
            a, out, n4 = stage_outside_fringe(src, a, out, B, K, protect_src)
    bdesc = B.round().astype(int).tolist() if B.ndim == 1 else mode
    kind = f"line art (outline {cover:.2f})" if lineart else f"photo/painting (outline {cover:.2f})"
    info = (f"{kind} | background {bdesc} | "
            f"edges {n1} px, slivers {n2}, ink/fill/paper {n3}, outside outline {n4}")
    return np.clip(out, 0, 255), a, info
