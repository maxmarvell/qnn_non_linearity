import matplotlib.pyplot as plt

tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 10,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.style.use("dark_background")
plt.style.use('seaborn-v0_8-darkgrid')
plt.rc('text.latex', preamble=R'\usepackage{amsmath} \usepackage{bbold}')
plt.rcParams.update(tex_fonts)

plt.rcParams['axes.axisbelow'] = True