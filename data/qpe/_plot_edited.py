import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from h2xh2.experiment import ChemData

matplotlib.rcParams["font.size"] = 12

# Chemistry data.
chem_data = ChemData()

ls_n_shots = [100, 500, 1000, 2300]

fig, ax = plt.subplots(
    nrows=4,
    ncols=1,
    sharex=True,
    figsize=(7, 6),
    dpi=300,
)

for i, n_shots in enumerate(ls_n_shots):
    with open(f"data_{n_shots:04d}.txt") as f:
        data = json.load(f)
    data = np.array(data)
    phi = data[:, 0]
    posterior_log = data[:, 1]

    phi2 = np.pi * phi
    phi_energy = -phi / chem_data.DELTAT

    ax[i].cla()
    ax[i].plot(phi2, posterior_log, "k-")
    ax[i].plot(
        [-np.pi * chem_data.FCI_ENERGY * chem_data.DELTAT] * 2,
        [min(posterior_log), max(posterior_log) * 0.9],
        "r--",
        label=r"$-E_{\mathrm{FCI}}t$",
    )
    ax[i].set_xlim(phi2[0], phi2[-1])
    ax[i].grid()
    if i == 3:
        ax[i].set_ylabel(r"$\log Q(\tilde{\phi})$", labelpad=2)
    else:
        ax[i].set_ylabel(r"$\log Q(\tilde{\phi})$", labelpad=10)
    ax[i].text(
        0.02,
        0.9,
        rf"$N_{{s}} = {n_shots}$",
        transform=ax[i].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
    )
ax[0].legend()
ax[3].set_xlabel(r"$\tilde{\phi}$")
plt.tight_layout()
# plt.show()

plt.savefig("_fig_iqpe.pdf", dpi=300, bbox_inches="tight")
