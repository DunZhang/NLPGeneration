import unicodedata
import re


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters

def normalizeString(s, is_ch=False):
    """

    :param s:
    :param is_ch: is chinese?
    :return:
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    if not is_ch:
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


if __name__ == "__main__":
    s = "hi你好啊 ci"
    print(normalizeString(s, True))
