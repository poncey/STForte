from __future__ import annotations
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


def rgb2hex(rgb_codes):
    hex_codes = []
    for rgb_code in rgb_codes:
        # Extracting RGB values
        rgb_values = rgb_code[rgb_code.index("(") + 1 : rgb_code.index(")")].split(",")
        red, green, blue = map(int, rgb_values)

        # Converting RGB to hexadecimal
        hex_code = f"#{red:02X}{green:02X}{blue:02X}"
        hex_codes.append(hex_code)
    return hex_codes


def hex2rgb(hex_codes: Union[str, list], csslike: bool = True):
    if not isinstance(hex_codes, list):
        hex_codes = [hex_codes]
    rgbcodes = []
    for hex_code in hex_codes:
        hex_code = hex_code.lstrip("#")  # Remove '#' if present
        if (
            len(hex_code) == 3
        ):  # Convert short HEX code to full length (e.g., #ABC to #AABBCC)
            hex_code = "".join(c * 2 for c in hex_code)

        r = int(hex_code[0:2], 16)
        g = int(hex_code[2:4], 16)
        b = int(hex_code[4:6], 16)
        if csslike:
            rgbcodes.append(f"rgb({r}, {g}, {b})")
        else:
            rgbcodes.append((r, g, b))
    return rgbcodes


def colormap_with_alpha(
    cmap: mpl.colors.LinearSegmentedColormap, alpha_scale: float = 20
):
    a_cmap = cmap(np.arange(cmap.N))
    xx = np.linspace(0, 1, cmap.N)
    yy = -np.exp(-alpha_scale * xx) + 1
    a_cmap[:, -1] = yy
    a_cmap = mpl.colors.ListedColormap(a_cmap)
    return a_cmap


def rgb_string_to_tuple(rgb_string):
    # Remove 'rgb(' and ')' from the string
    rgb_string = rgb_string.replace("rgb(", "").replace(")", "")

    # Split the string into individual components
    r, g, b = rgb_string.split(",")

    # Convert the components to integers and create a tuple
    rgb_tuple = (int(r), int(g), int(b))

    return rgb_tuple


def rgb_string_to_normalized_tuple(rgb_string):
    # Remove 'rgb(' and ')' from the string
    rgb_string = rgb_string.replace("rgb(", "").replace(")", "")

    # Split the string into individual components
    r, g, b = rgb_string.split(",")

    # Convert the components to integers and create a tuple
    r_normalized = int(r) / 255.0
    g_normalized = int(g) / 255.0
    b_normalized = int(b) / 255.0

    normalized_rgb_tuple = (r_normalized, g_normalized, b_normalized)

    return normalized_rgb_tuple


def create_refined_colormap(color_palette, stages: int = 256):
    colors_rgb = [rgb_string_to_normalized_tuple(color) for color in color_palette]
    positions = np.linspace(0, 1, len(colors_rgb))

    refined_positions = np.linspace(0, 1, stages)
    refined_colormap = []

    for i in range(3):  # Three channels: red, green, blue
        channel_colors = [color[i] for color in colors_rgb]
        interp_channel = np.interp(refined_positions, positions, channel_colors)
        refined_colormap.append(interp_channel)

    refined_colormap = np.transpose(refined_colormap)
    refined_colormap = tuple(map(tuple, refined_colormap))

    return LinearSegmentedColormap.from_list("RefinedColormap", refined_colormap)


# Color palettes
# from http://godsnotwheregodsnot.blogspot.de/2012/09/color-distribution-methodology.html
godsnot_102 = [
    # "#000000",  # remove the black, as often, we have black colored annotation
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#6A3A4C",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
]


# Alphabet colors generated from https://medialab.github.io/iwanthue/
iwanthue_alphabet_hard = [
    "#009480",
    "#fd325b",
    "#3ee26f",
    "#364fd4",
    "#8bd840",
    "#e387ff",
    "#bfd034",
    "#bf007f",
    "#00a656",
    "#ac0036",
    "#00c9a3",
    "#9f2600",
    "#699cff",
    "#cca400",
    "#aeaaff",
    "#e16d00",
    "#005d91",
    "#fdb879",
    "#8f2a6b",
    "#0a6300",
    "#ff9dcd",
    "#606000",
    "#71d3f4",
    "#ffa289",
    "#a15c65",
    "#cb8285",
]

# 42 colors generated from https://medialab.github.io/iwanthue/
iwanthue_answer_hard = [
    "#eb9300",
    "#9c61eb",
    "#189c0f",
    "#c857dd",
    "#80db6b",
    "#6436b1",
    "#f8bc2a",
    "#015ec2",
    "#c9cd4e",
    "#c60098",
    "#00b36c",
    "#f23bb0",
    "#017428",
    "#eb0067",
    "#00865d",
    "#ff529f",
    "#376200",
    "#8889ff",
    "#ac9700",
    "#0285d3",
    "#da6b00",
    "#00c9f1",
    "#db4615",
    "#01adde",
    "#f14536",
    "#6cc8ff",
    "#a15a00",
    "#b3bfff",
    "#a31432",
    "#93d5a3",
    "#a00d5c",
    "#008d7d",
    "#ff78c6",
    "#c3cc88",
    "#882d7a",
    "#ff9a67",
    "#73a3d4",
    "#7f4327",
    "#ff9bba",
    "#7a4366",
    "#d5a77d",
    "#8e3248",
]

# 102 colors generated from https://medialab.github.io/iwanthue/
iwanthue_102_hard = [
    "#6c3dc2",
    "#a1cf27",
    "#3e4bcf",
    "#43a100",
    "#9738bf",
    "#18d666",
    "#b5009b",
    "#36af29",
    "#9360ea",
    "#8cda51",
    "#1261e8",
    "#ecc22c",
    "#6376ff",
    "#c8b000",
    "#7184ff",
    "#4c9500",
    "#eb7bff",
    "#008812",
    "#e65cdd",
    "#007500",
    "#e23cb9",
    "#009c43",
    "#f52793",
    "#66dd8c",
    "#d4007b",
    "#00a869",
    "#f10d6e",
    "#02ba96",
    "#d10049",
    "#38dade",
    "#b31800",
    "#01c2dc",
    "#ff5f36",
    "#026ad0",
    "#e8a200",
    "#5d3daa",
    "#dcc740",
    "#ad81ff",
    "#638600",
    "#dd90ff",
    "#9b9500",
    "#911b88",
    "#c2ce67",
    "#0158ae",
    "#fb8918",
    "#56a4ff",
    "#ff742f",
    "#0282b6",
    "#c96600",
    "#6fd2f8",
    "#b20018",
    "#8cd5b0",
    "#cc0062",
    "#00722e",
    "#ff67c0",
    "#4c6800",
    "#be9fff",
    "#a28b00",
    "#cdafff",
    "#c37700",
    "#b9baff",
    "#aa7300",
    "#f8acfd",
    "#3e5b0e",
    "#ff70b9",
    "#01754c",
    "#ff4e61",
    "#0f5e44",
    "#ff547a",
    "#bccf77",
    "#84307d",
    "#d3c86e",
    "#4a4d87",
    "#e5c367",
    "#773f6b",
    "#fdb96b",
    "#9595cb",
    "#ff6147",
    "#3f5a26",
    "#ff77a9",
    "#686b00",
    "#c40030",
    "#56551e",
    "#ff8087",
    "#685001",
    "#c88fb7",
    "#895e00",
    "#88345f",
    "#f0bc88",
    "#932e43",
    "#ffb2a2",
    "#9e231a",
    "#eaa791",
    "#9a3a00",
    "#e398a0",
    "#933130",
    "#a48355",
    "#ff9370",
    "#9a565a",
    "#ff9f91",
    "#865838",
    "#a36854",
]

# 32 colors generated by kmeans
iwanthue_32_soft = [
    "#4baebc",
    "#e0462a",
    "#593ccf",
    "#60bf37",
    "#a83bd7",
    "#4eba68",
    "#d851ce",
    "#aeb538",
    "#4d2992",
    "#5d7e2c",
    "#816be1",
    "#d67a31",
    "#5077cd",
    "#cc9e4e",
    "#8f429d",
    "#5bb18e",
    "#db4092",
    "#36613e",
    "#d73a5b",
    "#77a0d0",
    "#973727",
    "#b48bd0",
    "#484218",
    "#d679af",
    "#a0ae72",
    "#46346f",
    "#8d6337",
    "#585f85",
    "#db7670",
    "#8e3265",
    "#cb9298",
    "#6c2f39",
]

# Prism Light palette from GraphPad Prism
prism_light_palette = [
    "#A48AD3",
    "#1CC5FE",
    "#6FC7CF",
    "#FBA27D",
    "#FB7D80",
    "#2C1453",
    "#114CE8",
    "#0E6F7C",
    "#FB4F06",
    "#FB0005",
]

# Prism Dark palette from GraphPad Prism
prism_dark_palette = [
    "#2C1453",
    "#114CE8",
    "#0E6F7C",
    "#FB4F06",
    "#FB0005",
    "#A48AD3",
    "#1CC5FE",
    "#6FC7CF",
    "#FBA27D",
    "#FB7D80",
]

# 1960s palette from GraphPad Prism
prism_1960s_palette = [
    "#7BB44F",
    "#15A6EC",
    "#F5C02C",
    "#EE2926",
    "#961192",
    "#16325C",
]

# 2000s palette from GraphPad Prism
prism_2000s_palette = [
    "#155BE4",
    "#28CE53",
    "#FF0000",
    "#BE1572",
    "#1D8DEE",
    "#6AA823",
    "#FC4B08",
    "#000000",
]

accent_palette = [
    "#7fc97f",
    "#beaed4",
    "#fdc086",
    "#ffff99",
    "#386cb0",
    "#f0027f",
    "#bf5b17",
    "#666666",
]
