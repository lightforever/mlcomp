import io
from PIL import Image
from matplotlib.figure import Figure


def figure_to_binary(figure: Figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    content = buf.read()
    buf.close()
    return content


def figure_to_img(figure: Figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    im = Image.open(buf)
    im.show()
    buf.close()
    return im