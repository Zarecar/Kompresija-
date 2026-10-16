"""ffmpeg kodeci bez gubitaka kao osnovne linije (metode 0x20-0x2F). Deo referentne mašine M.

FORMAT ZAPISA (samoograničavajući, kao i svi zapisi KFB)
    [m][meta: 8 varint-ova][varint LV][video tok: LV B][varint A][audio tok: A >> 2 B]

  m      metoda = video kodek: 0x20 FFV1, 0x21 H.264 (x264rgb), 0x22 VP9, 0x23 AV1,
         0x24 APNG, 0x25 HEVC (x265); 0x26-0x2F rezervisano (greška)
  A & 3  način kodovanja zvuka; A >> 2 je dužina audio toka u bajtovima:
           0  tišina: nema toka (dužina mora biti 0), svi uzorci su nula
           1  sirovi PCM: tačno meta.audio_bytes bajtova, kao u payload-u
           2  FLAC (samo sample_bytes = 2, 1..8 kanala, 1 <= audio_rate < 2^20):
              niz FLAC frejmova BEZ zaglavlja; dekoder dodaje "fLaC" + STREAMINFO
              izveden iz metapodataka (blok 16..65535, veličine frejmova i MD5
              nepoznati, 16 bita, audio_rate, audio_channels, audio_samples) i
              dekoduje ga u s16le
           3  rezervisano (greška)
  LV = 0 je dozvoljeno samo ako je meta.video_bytes = 0 (nema slike).

VIDEO TOK po načinu uokvirivanja kodeka:
  "pkt"  (FFV1, VP9) [varint E][E B extradata][varint P][P varint-ova: veličine
         paketa][paketi]. Dekoder od toga sastavlja minimalan Matroska fajl: EBML
         zaglavlje (DocType matroska), Info (TimecodeScale 1 ms), jedan TrackEntry
         (broj 1, CodecID V_FFV1 / V_VP9, CodecPrivate = extradata ako E > 0,
         PixelWidth/Height iz meta) i jedan Cluster sa P SimpleBlock-ova (paket i
         ima oznaku i ms, zastavica 0x80) -> ffmpeg -f matroska.
  "raw"  elementarni tok koji ffmpeg sam deli na pakete: H.264 i HEVC Annex B
         (bez SEI/AUD, početni kodovi od 3 B), AV1 "low overhead" OBU (Section 5,
         -f obu), APNG fajl (-f apng).
Dekodovanje: ffmpeg CLI sa fiksnim demuxer-om i dekoderom (ffv1, h264, vp9,
libdav1d, apng, hevc), 1 nit, -xerror, -fps_mode passthrough, izlaz rawvideo
rgb24. Izlaz mora imati TAČNO T*H*W*3 bajtova (inače greška). Filter showinfo
(prvi u lancu, pre svake konverzije) prijavljuje SVAKI dekodovani frejm: mora ih
biti tačno T, svaki dimenzija W x H iz meta i u formatu iz RGB_FMTS (8-bitni RGB,
konverzija u rgb24 je samo izbor/permutacija bajtova). Tok koji se dekoduje u
YUV/sivo ili menja dimenzije je greška: takva konverzija (swscale) zavisi od
SIMD putanje procesora, pa izlaz ne bi bio isti na svakoj mašini. Isto važi za
FLAC (ashowinfo): svaki frejm s16/s16p sa audio_channels i audio_rate iz meta
(inače bi ffmpeg tiho ubacio resampler). Varint-ovi zaglavlja imaju najviše
10 bajtova. Sve ide kroz privremeni direktorijum, sa vremenskim ograničenjem i
ograničenjem veličine.

Enkoderska strana (encode_video/encode_audio/pack i encode) nije normativna; ona
samo pravi tokove u ovom formatu. Pretraga podešavanja je u kfb/encoders/ffmpeg_enc.py.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from igc.film import Meta, get_varint, put_varint

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
MAX_PAYLOAD = 1 << 30            # zaštita memorije pri dekodovanju
MAX_DIM = 1 << 14
A_SILENCE, A_RAW, A_FLAC = 0, 1, 2
_BASE = [FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
# dekodovanje: 1 nit, greška prekida; nivo "info" bez statistike zbog linija (a)showinfo
_DEC = [FFMPEG, "-nostdin", "-hide_banner", "-nostats", "-loglevel", "info", "-y", "-xerror", "-threads", "1"]
_BITEXACT = ["-fflags", "+bitexact", "-flags", "+bitexact", "-map_metadata", "-1"]
# formati dekodovanog frejma čija je konverzija u rgb24 samo izbor/permutacija bajtova
RGB_FMTS = frozenset({b"gbrp", b"bgr0", b"rgb24", b"rgba", b"bgra"})
A_FMTS = frozenset({b"s16", b"s16p"})
# linije filtera showinfo/ashowinfo (av_log prefiks je uvek na početku linije; metapodaci toka
# se ispisuju uvučeno, pa ne mogu da podmetnu ovakvu liniju)
_VINFO = re.compile(rb"^\[Parsed_showinfo_0 @ 0x[0-9a-f]+\] n: *\d+ [^\n]*? fmt:([0-9a-z]+) [^\n]*? s:(\d+)x(\d+) ",
                    re.M)
_AINFO = re.compile(rb"^\[Parsed_ashowinfo_0 @ 0x[0-9a-f]+\] n:\d+ [^\n]*? fmt:([0-9a-z]+) channels:(\d+) "
                    rb"[^\n]*? rate:(\d+) ", re.M)


X265_PARAMS = "lossless=1:subme=0:log-level=error:info=0:frame-threads=1:pools=none"


@dataclass(frozen=True)
class Codec:
    name: str
    framing: str                 # "pkt" (Matroska omotač u dekoderu) ili "raw"
    demux: str                   # ffmpeg demuxer pri dekodovanju
    decoder: str                 # ffmpeg dekoder
    mkv_id: str = ""             # CodecID za "pkt"
    enc_args: tuple = ()         # podrazumevano kodovanje (encode); pretraga je u enkoderu
    enc_fmt: str = ""            # format izlaza ffmpeg-a pri kodovanju
    keep_extradata: bool = True


CODECS: dict[int, Codec] = {
    0x20: Codec("ffv1", "pkt", "matroska", "ffv1", "V_FFV1",
                ("-pix_fmt", "bgr0", "-c:v", "ffv1", "-level", "3", "-coder", "1", "-context", "1",
                 "-g", "9999", "-slices", "1", "-slicecrc", "0", "-threads", "1"), "matroska"),
    0x21: Codec("x264rgb", "raw", "h264", "h264", "",
                ("-pix_fmt", "rgb24", "-c:v", "libx264rgb", "-qp", "0", "-preset", "veryslow",
                 "-threads", "1", "-bsf:v", "filter_units=remove_types=6"), "h264"),
    0x22: Codec("vp9", "pkt", "matroska", "vp9", "V_VP9",
                ("-pix_fmt", "gbrp", "-c:v", "libvpx-vp9", "-lossless", "1", "-deadline", "good",
                 "-cpu-used", "4", "-g", "9999", "-threads", "1", "-row-mt", "0", "-tile-columns", "0"),
                "matroska", keep_extradata=False),
    0x23: Codec("av1", "raw", "obu", "libdav1d", "",
                ("-pix_fmt", "gbrp", "-c:v", "libaom-av1", "-aom-params", "lossless=1", "-cpu-used", "6",
                 "-g", "9999", "-threads", "1", "-row-mt", "0", "-tiles", "1x1", "-lag-in-frames", "0"),
                "obu"),
    0x24: Codec("apng", "raw", "apng", "apng", "",
                ("-pix_fmt", "rgb24", "-c:v", "apng", "-pred", "mixed", "-compression_level", "9",
                 "-plays", "0"), "apng"),
    0x25: Codec("x265", "raw", "hevc", "hevc", "",
                ("-pix_fmt", "gbrp", "-c:v", "libx265", "-preset", "veryslow",
                 "-x265-params", X265_PARAMS,
                 "-bsf:v", "filter_units=remove_types=35|39|40"), "hevc"),
}
METHODS: dict[str, int] = {c.name: m for m, c in CODECS.items()}


# ------------------------------------------------------------------ pomoćne
def _run(cmd: list, timeout: float) -> bytes:
    """Pokreće ffmpeg; vraća stderr (linije showinfo pri dekodovanju), greška -> ValueError."""
    try:
        p = subprocess.run([str(c) for c in cmd], stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=max(1.0, timeout))
    except subprocess.TimeoutExpired:
        raise ValueError("ffmpeg: isteklo vreme") from None
    except OSError as e:
        raise ValueError(f"ffmpeg nije dostupan: {e}") from None
    if p.returncode != 0:
        raise ValueError("ffmpeg: " + p.stderr.decode("utf-8", "replace").strip()[-300:])
    return p.stderr


def _varint(d: bytes, pos: int) -> tuple[int, int]:
    """get_varint ograničen na 10 bajtova: dugačak niz 0xFF bi se inače parsirao kvadratno."""
    head = d[pos:pos + 10]
    if len(head) == 10 and all(b & 0x80 for b in head):
        raise ValueError("predugačak varint")
    return get_varint(d, pos)


def _check_meta(meta: Meta) -> None:
    if meta.payload_bytes > MAX_PAYLOAD or max(meta.width, meta.height) > MAX_DIM:
        raise ValueError("film je prevelik za ffmpeg metode")


def flac_ok(meta: Meta) -> bool:
    """FLAC (način 2) je definisan samo za 16-bitni zvuk sa 1..8 kanala."""
    return (meta.sample_bytes == 2 and 1 <= meta.audio_channels <= 8 and 1 <= meta.audio_rate < (1 << 20)
            and 0 < meta.audio_samples < (1 << 36))


def _decode_timeout(nbytes: int) -> float:
    return 120.0 + nbytes / 20000.0


# ------------------------------------------------------------ Matroska (EBML)
def _eid(i: int) -> bytes:
    return i.to_bytes((i.bit_length() + 7) // 8, "big")


def _el(i: int, data: bytes) -> bytes:
    return _eid(i) + b"\x01" + len(data).to_bytes(7, "big") + data


def _uint(i: int, v: int) -> bytes:
    return _el(i, v.to_bytes(max(1, (v.bit_length() + 7) // 8), "big"))


def mkv_wrap(codec_id: str, meta: Meta, extradata: bytes, packets: list[bytes]) -> bytes:
    """Minimalan Matroska fajl sa jednim video tragom (deo dekodera)."""
    if len(packets) > 32767:
        raise ValueError("previše paketa")
    head = _el(0x1A45DFA3, _uint(0x4286, 1) + _uint(0x42F7, 1) + _uint(0x42F2, 4) + _uint(0x42F3, 8)
               + _el(0x4282, b"matroska") + _uint(0x4287, 4) + _uint(0x4285, 2))
    info = _el(0x1549A966, _uint(0x2AD7B1, 1000000))
    video = _el(0xE0, _uint(0xB0, meta.width) + _uint(0xBA, meta.height))
    te = (_uint(0xD7, 1) + _uint(0x73C5, 1) + _uint(0x83, 1) + _el(0x86, codec_id.encode())
          + (_el(0x63A2, extradata) if extradata else b"") + video)
    tracks = _el(0x1654AE6B, _el(0xAE, te))
    blocks = b"".join(_el(0xA3, b"\x81" + i.to_bytes(2, "big") + b"\x80" + p) for i, p in enumerate(packets))
    cluster = _el(0x1F43B675, _uint(0xE7, 0) + blocks)
    return head + _el(0x18538067, info + tracks + cluster)


def _vint(d: bytes, pos: int, keep_marker: bool) -> tuple[int, int, bool]:
    if pos >= len(d):
        raise ValueError("skraćen EBML")
    b = d[pos]
    n = 1
    while n <= 8 and not b & (0x80 >> (n - 1)):
        n += 1
    if n > 8 or pos + n > len(d):
        raise ValueError("neispravan EBML vint")
    v = b if keep_marker else b & (0xFF >> n)
    for k in range(1, n):
        v = (v << 8) | d[pos + k]
    unknown = not keep_marker and v == (1 << (7 * n)) - 1
    return v, pos + n, unknown


def mkv_parse(d: bytes) -> tuple[str, bytes, list[bytes]]:
    """(CodecID, CodecPrivate, paketi) prvog traga iz Matroska fajla (enkoderska strana)."""
    MASTER = {0x18538067, 0x1654AE6B, 0xAE, 0x1F43B675, 0xA0}
    codec_id, priv, pkts = "", b"", []
    stack = [len(d)]
    pos = 0
    while pos < len(d):
        while stack and pos >= stack[-1]:
            stack.pop()
        eid, p2, _ = _vint(d, pos, True)
        size, p3, unknown = _vint(d, p2, False)
        end = stack[-1] if (unknown or p3 + size > stack[-1]) else p3 + size
        if eid in MASTER:
            stack.append(end)
            pos = p3
            continue
        body = d[p3:end]
        if eid == 0x86 and not codec_id:
            codec_id = body.decode("ascii", "replace")
        elif eid == 0x63A2 and not priv:
            priv = body
        elif eid in (0xA3, 0xA1):
            tn, q, _ = _vint(body, 0, False)
            if tn == 1:
                if body[q + 2] & 0x06:
                    raise ValueError("lacing nije podržan")
                pkts.append(body[q + 3:])
        pos = end
    return codec_id, priv, pkts


def pack_packets(extradata: bytes, packets: list[bytes]) -> bytes:
    buf = bytearray()
    put_varint(buf, len(extradata))
    buf += extradata
    put_varint(buf, len(packets))
    for p in packets:
        put_varint(buf, len(p))
    for p in packets:
        buf += p
    return bytes(buf)


def unpack_packets(vs: bytes, max_packets: int) -> tuple[bytes, list[bytes]]:
    e, pos = _varint(vs, 0)
    if pos + e > len(vs):
        raise ValueError("skraćen extradata")
    extra = vs[pos:pos + e]
    pos += e
    n, pos = _varint(vs, pos)
    if not 1 <= n <= max_packets:
        raise ValueError("neispravan broj paketa")
    sizes = []
    for _ in range(n):
        s, pos = _varint(vs, pos)
        if s < 1:
            raise ValueError("prazan paket")
        sizes.append(s)
    if pos + sum(sizes) != len(vs):
        raise ValueError("veličine paketa ne odgovaraju toku")
    pkts = []
    for s in sizes:
        pkts.append(vs[pos:pos + s])
        pos += s
    return extra, pkts


# ------------------------------------------------------------------ FLAC
def flac_header(meta: Meta) -> bytes:
    """'fLaC' + STREAMINFO (poslednji blok) izveden iz metapodataka (deo dekodera)."""
    si = (16).to_bytes(2, "big") + (65535).to_bytes(2, "big") + bytes(6)
    v = (meta.audio_rate << 44) | ((meta.audio_channels - 1) << 41) | (15 << 36) | meta.audio_samples
    si += v.to_bytes(8, "big") + bytes(16)
    return b"fLaC" + bytes([0x80, 0, 0, len(si)]) + si


def flac_frames(flac: bytes, meta: Meta) -> bytes:
    """Iz FLAC fajla izdvaja frejmove i proverava da STREAMINFO odgovara metapodacima."""
    if flac[:4] != b"fLaC":
        raise ValueError("nije FLAC")
    pos, si = 4, None
    while True:
        h = flac[pos]
        ln = int.from_bytes(flac[pos + 1:pos + 4], "big")
        if h & 0x7F == 0:
            si = flac[pos + 4:pos + 4 + ln]
        pos += 4 + ln
        if h & 0x80:
            break
    if si is None or si[10:18] != flac_header(meta)[18:26]:
        raise ValueError("STREAMINFO ne odgovara metapodacima")
    return flac[pos:]


# ------------------------------------------------------------------ dekoder
def _decode_video(codec: Codec, meta: Meta, vs: bytes, tmp: Path) -> bytes:
    n = meta.video_bytes
    if n == 0:
        if vs:
            raise ValueError("video tok bez slike")
        return b""
    if not vs:
        raise ValueError("nedostaje video tok")
    if codec.framing == "pkt":
        extra, pkts = unpack_packets(vs, 4 * meta.frames + 16)
        blob = mkv_wrap(codec.mkv_id, meta, extra, pkts)
    else:
        blob = vs
    src, dst = tmp / "v.in", tmp / "v.rgb"
    src.write_bytes(blob)
    # max_pixels: zaštita od toka čije su dimenzije mnogo veće od metapodataka
    err = _run(_DEC + ["-max_pixels", (meta.width + 256) * (meta.height + 256), "-f", codec.demux,
                       "-c:v", codec.decoder, "-i", src, "-map", "0:v:0", "-an", "-sn", "-dn",
                       "-vf", "showinfo=checksum=0", "-fps_mode", "passthrough", "-frames:v", meta.frames + 1,
                       "-pix_fmt", "rgb24", "-f", "rawvideo", "-fs", n + 1, dst], _decode_timeout(n))
    out = dst.read_bytes() if dst.exists() else b""
    if len(out) != n:
        raise ValueError(f"video se dekoduje u {len(out)} B umesto {n} B")
    frames = _VINFO.findall(err)
    if len(frames) != meta.frames:
        raise ValueError(f"dekodovano {len(frames)} frejmova umesto {meta.frames}")
    for fmt, w, h in frames:
        if fmt not in RGB_FMTS or (int(w), int(h)) != (meta.width, meta.height):
            raise ValueError(f"dekodovan frejm {fmt.decode()} {int(w)}x{int(h)}: dozvoljen je samo "
                             f"8-bitni RGB {meta.width}x{meta.height}")
    return out


def _decode_audio(meta: Meta, mode: int, data: bytes, tmp: Path) -> bytes:
    n = meta.audio_bytes
    if mode == A_SILENCE:
        if data:
            raise ValueError("tišina ne sme imati tok")
        return bytes(n)
    if mode == A_RAW:
        if len(data) != n:
            raise ValueError("pogrešna dužina sirovog zvuka")
        return data
    if mode == A_FLAC:
        if not flac_ok(meta) or not data:
            raise ValueError("FLAC nije primenljiv")
        src, dst = tmp / "a.flac", tmp / "a.pcm"
        src.write_bytes(flac_header(meta) + data)
        err = _run(_DEC + ["-f", "flac", "-c:a", "flac", "-i", src, "-map", "0:a:0", "-vn", "-af", "ashowinfo",
                           "-c:a", "pcm_s16le", "-f", "s16le", "-fs", n + 1, dst], _decode_timeout(n))
        out = dst.read_bytes() if dst.exists() else b""
        if len(out) != n:
            raise ValueError(f"FLAC se dekoduje u {len(out)} B umesto {n} B")
        afr = _AINFO.findall(err)
        if not afr or any(f not in A_FMTS or int(c) != meta.audio_channels or int(r) != meta.audio_rate
                          for f, c, r in afr):
            raise ValueError("FLAC frejmovi ne odgovaraju metapodacima (format, kanali ili učestanost)")
        return out
    raise ValueError(f"nepoznat način zvuka {mode}")


def decode(rec: bytes) -> tuple[Meta, bytes, int]:
    if not rec or rec[0] not in CODECS:
        raise ValueError("nepoznata ffmpeg metoda")
    codec = CODECS[rec[0]]
    pos = 1
    for _ in range(8):                      # ograničeni varint-ovi pre igc.film.Meta.from_bytes
        _, pos = _varint(rec, pos)
    meta, pos = Meta.from_bytes(rec, 1)
    _check_meta(meta)
    lv, pos = _varint(rec, pos)
    if pos + lv > len(rec):
        raise ValueError("skraćen video tok")
    vs = rec[pos:pos + lv]
    pos += lv
    a, pos = _varint(rec, pos)
    mode, la = a & 3, a >> 2
    if mode > A_FLAC:
        raise ValueError(f"nepoznat način zvuka {mode}")
    if pos + la > len(rec):
        raise ValueError("skraćen audio tok")
    data = rec[pos:pos + la]
    pos += la
    with tempfile.TemporaryDirectory(prefix="kfb_ffdec_") as d:
        video = _decode_video(codec, meta, vs, Path(d))
        audio = _decode_audio(meta, mode, data, Path(d))
    return meta, video + audio, pos


# ------------------------------------------------------------ enkoderska strana
def pack(method_id: int, meta: Meta, vstream: bytes, amode: int, astream: bytes) -> bytes:
    buf = bytearray([method_id])
    buf += meta.to_bytes()
    put_varint(buf, len(vstream))
    buf += vstream
    put_varint(buf, (len(astream) << 2) | amode)
    buf += astream
    return bytes(buf)


def _short_startcodes(es: bytes) -> bytes:
    # u NAL telu nema 00 00 00 (emulation prevention), pa je svako 00 00 00 01 početni kod
    return es.replace(b"\x00\x00\x00\x01", b"\x00\x00\x01")


def encode_video(method_id: int, meta: Meta, video: bytes, args=None, timeout: float = 600.0,
                 tmp: Path | None = None) -> bytes | None:
    """Video tok (u formatu metode) za date ffmpeg argumente kodera, ili None."""
    codec = CODECS[method_id]
    if meta.video_bytes == 0:
        return b""
    args = list(codec.enc_args if args is None else args)
    with tempfile.TemporaryDirectory(prefix="kfb_ffenc_", dir=tmp) as d:
        d = Path(d)
        src, dst = d / "v.rgb", d / f"v.{codec.enc_fmt}"
        src.write_bytes(video)
        try:
            _run(_BASE + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{meta.width}x{meta.height}",
                          "-r", max(1, meta.fps), "-i", src] + _BITEXACT + ["-an"] + args
                 + ["-f", codec.enc_fmt, dst], timeout)
            out = dst.read_bytes()
        except (ValueError, OSError):
            return None
    if not out:
        return None
    if codec.framing == "pkt":
        try:
            cid, priv, pkts = mkv_parse(out)
        except (ValueError, IndexError):
            return None
        if not pkts:
            return None
        extra = priv[40:] if cid == "V_MS/VFW/FOURCC" else priv
        return pack_packets(extra if codec.keep_extradata else b"", pkts)
    if codec.demux in ("h264", "hevc"):
        return _short_startcodes(out)
    return out


def encode_audio(meta: Meta, audio: bytes, args=None, timeout: float = 120.0,
                 tmp: Path | None = None) -> tuple[int, bytes]:
    """Najkraći audio tok: tišina, FLAC (ako je kraći) ili sirovi PCM."""
    if not any(audio):
        return A_SILENCE, b""
    best = (A_RAW, audio)
    if flac_ok(meta):
        args = list(args or ["-compression_level", "12"])
        with tempfile.TemporaryDirectory(prefix="kfb_ffenc_", dir=tmp) as d:
            src, dst = Path(d) / "a.pcm", Path(d) / "a.flac"
            src.write_bytes(audio)
            try:
                _run(_BASE + ["-f", "s16le", "-ar", meta.audio_rate, "-ac", meta.audio_channels, "-i", src]
                     + _BITEXACT + ["-c:a", "flac"] + args + ["-f", "flac", dst], timeout)
                fr = flac_frames(dst.read_bytes(), meta)
                if fr and len(fr) < len(audio):
                    best = (A_FLAC, fr)
            except (ValueError, OSError, IndexError):
                pass
    return best


def encode(method_id: int, meta: Meta, payload: bytes) -> bytes | None:
    """Kompletan zapis sa podrazumevanim podešavanjem kodeka; None ako nije tačan."""
    if method_id not in CODECS or len(payload) != meta.payload_bytes:
        return None
    vs = encode_video(method_id, meta, payload[:meta.video_bytes])
    if vs is None:
        return None
    rec = pack(method_id, meta, vs, *encode_audio(meta, payload[meta.video_bytes:]))
    try:
        if decode(rec)[1] == payload:
            return rec
    except ValueError:
        pass
    return None
