from moshi.client_utils import colorize


def get_color_code(color_name: str) -> tuple:
    if color_name == "blue":
        return (92, 158, 255)
    elif color_name == "yellow":
        return (255, 255, 85)
    elif color_name == "red":
        return (255, 85, 85)
    elif color_name == "orange":
        return (255, 171, 64)
    elif color_name == "green":
        return (57, 242, 174)
    

def colorize_rgb(text: str, rgb: tuple) -> str:
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\033[0m"


def make_colored_log(level: str, msg: str, colored_parts: list[tuple[str, str]] | None = None) -> str:
    if level == "info":
        prefix = colorize_rgb("[Info]", get_color_code("blue"))
    elif level == "warning":
        prefix = colorize_rgb("[Warn]", get_color_code("yellow"))
    elif level == "error":
        prefix = colorize_rgb("[Err ]",  get_color_code("red"))
    else:
        raise ValueError(f"Unknown level {level}")
    if colored_parts is not None:
        msg = msg.format(*[colorize_rgb(text, get_color_code(color_code)) for text, color_code in colored_parts])
    return prefix + " " + msg


def log(level: str, msg: str, colored_parts: list[tuple[str, str]] | None = None) -> None:
    """Log something with a given level."""
    print(make_colored_log(level, msg, colored_parts))