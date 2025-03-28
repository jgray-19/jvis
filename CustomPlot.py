import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import Optional, Any


class CustomPlot:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[int, int] = (10, 6),
        style: Optional[str] = None,
    ) -> None:
        """
        Initialize a CustomPlot instance with a grid of subplots.

        Args:
            nrows (int): Number of subplot rows.
            ncols (int): Number of subplot columns.
            figsize (tuple): Size of the figure.
            style (str, optional): Matplotlib style to apply.
        """
        if style:
            plt.style.use(style)
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        # Ensure axes is always a 2D list even for a single subplot or 1D array.
        if not hasattr(self.axes, "__iter__"):
            self.axes = [[self.axes]]
        elif isinstance(self.axes, np.ndarray) and self.axes.ndim == 1:
            self.axes = [list(self.axes)]
        self.nrows: int = len(self.axes)
        self.ncols: int = len(self.axes[0])
        self.apply_custom_style()
        # Storage for plotted data: keys are (row, col) tuples.
        self.plotted_data: dict[tuple[int, int], list[dict]] = {}

    def apply_custom_style(self):
        plt.rcParams.update(
            {
                "font.size": 16,
                "axes.prop_cycle": plt.cycler(
                    color=[
                        "#0072B2",
                        "#D55E00",
                        "#009E73",
                        "#56B4E9",
                        "#E69F00",
                        "#F0E442",
                        "#CC79A7",
                        "#000000",
                    ]
                ),
                "grid.linestyle": "--",
                "grid.alpha": 0.6,
                "legend.fontsize": 12,
                "axes.titleweight": "bold",
            }
        )
        for row in self.axes:
            for ax in row:
                ax.grid(True)

    def get_axis(self, position: tuple[int, int]) -> plt.Axes:
        """
        Retrieve the axis at the specified (row, col) position.

        Args:
            position (tuple[int, int]): Tuple containing row and column indices.

        Returns:
            The matplotlib axes object.

        Raises:
            IndexError: If row or col is out of range.
        """
        row, col = position
        if row < 0 or row >= self.nrows:
            raise IndexError(f"Row index {row} out of range (0 to {self.nrows - 1}).")
        if col < 0 or col >= self.ncols:
            raise IndexError(
                f"Column index {col} out of range (0 to {self.ncols - 1})."
            )
        return self.axes[row][col]

    @property
    def flat_axes(self) -> list[Any]:
        """Return a flat list of all axes."""
        return [ax for row in self.axes for ax in row]

    def add_line(self, position: tuple[int, int], x, y, label=None, **kwargs):
        (line,) = self.get_axis(position).plot(x, y, label=label, **kwargs)

        # Store the data in the dictionary.
        if position not in self.plotted_data:
            self.plotted_data[position] = []

        self.plotted_data[position].append(
            {
                "type": "line",
                "x": np.asarray(x),
                "y": np.asarray(y),
                "label": label,
                "line_obj": line,
            }
        )

    def add_errorbar(
        self, position: tuple[int, int], x, y, yerr=None, label=None, **kwargs
    ):
        row, col = position
        ax = self.axes[row][col]
        errorbar_obj = ax.errorbar(x, y, yerr=yerr, label=label, **kwargs)
        if position not in self.plotted_data:
            self.plotted_data[position] = []
        self.plotted_data[position].append(
            {
                "type": "errorbar",
                "x": np.asarray(x),
                "y": np.asarray(y),
                "yerr": np.asarray(yerr) if yerr is not None else None,
                "label": label,
                "errorbar_obj": errorbar_obj,
            }
        )

    def set_all_labels(
        self,
        xlabel,
        ylabel,
        share_x=True,
        share_y=True,
        fontweight="bold",
        fontsize=16,
        color="black",
        pad=4,
    ):
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axes[i][j]
                if not share_x or (share_x and i == self.nrows - 1):
                    ax.set_xlabel(
                        xlabel,
                        fontweight=fontweight,
                        fontsize=fontsize,
                        color=color,
                        labelpad=pad,
                    )
                else:
                    ax.set_xlabel("")
                if not share_y or (share_y and j == 0):
                    ax.set_ylabel(
                        ylabel,
                        fontweight=fontweight,
                        fontsize=fontsize,
                        color=color,
                        labelpad=pad,
                    )
                else:
                    ax.set_ylabel("")

    def set_global_axis_limits(self, xlims, ylims):
        for row in self.axes:
            for ax in row:
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)

    def set_axis_limits_by_row(self, row_index, xlims=None, ylims=None):
        for ax in self.axes[row_index]:
            if xlims is not None:
                ax.set_xlim(xlims)
            if ylims is not None:
                ax.set_ylim(ylims)

    def set_axis_limits_by_col(self, col_index, xlims=None, ylims=None):
        for i in range(self.nrows):
            ax = self.axes[i][col_index]
            if xlims is not None:
                ax.set_xlim(xlims)
            if ylims is not None:
                ax.set_ylim(ylims)

    def remove_duplicate_ticks(self, share_x=True, share_y=True):
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axes[i][j]
                if share_x and i < self.nrows - 1:
                    ax.set_xticklabels([])
                    ax.tick_params(axis="x", which="both", bottom=False)
                if share_y and j > 0:
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", which="both", left=False)

    def format_legend(
        self,
        position: tuple[int, int],
        loc="best",
        ncol=1,
        frameon=True,
        **kwargs,
    ):
        row, col = position
        self.axes[row][col].legend(loc=loc, ncol=ncol, frameon=frameon, **kwargs)

    def apply_tick_format(self, position: tuple[int, int], xformat=None, yformat=None):
        row, col = position
        if xformat:
            self.axes[row][col].xaxis.set_major_formatter(
                mticker.FormatStrFormatter(xformat)
            )
        if yformat:
            self.axes[row][col].yaxis.set_major_formatter(
                mticker.FormatStrFormatter(yformat)
            )

    def add_secondary_axis_with_difference(
        self,
        position: tuple[int, int],
        x,
        y1,
        y2,
        diff_label="Difference",
        diff_color="gray",
        match_color=True,
        labelpad=20,
        **kwargs,
    ):
        """
        Create a twin y-axis on the specified subplot that plots the difference y1 - y2.

        Parameters:
            position (tuple[int, int]): Tuple containing row and column indices.
            x (array-like): X values.
            y1 (array-like): First data series.
            y2 (array-like): Second data series (to subtract from y1).
            diff_label (str): Y-axis label for the difference axis.
            diff_color (str): Colour for the line and optionally the axis/ticks.
            match_color (bool): If True, axis line, ticks, and labels are coloured the same as diff_color.
            labelpad (int): Padding between the axis and its label.
            label_offset (tuple): (x, y) coordinates for the label in axis relative coordinates.
                                Adjust these to move the label away from the ticks.
            **kwargs: Additional arguments passed to ax.plot().

        Returns:
            twin_ax: The secondary y-axis.
        """
        import numpy as np

        row, col = position
        ax = self.axes[row][col]
        twin_ax = ax.twinx()

        # Compute the difference and plot it.
        diff = np.asarray(y1) - np.asarray(y2)
        (line,) = twin_ax.plot(x, diff, label=diff_label, color=diff_color, **kwargs)

        # Compute label text.
        avg_diff = np.mean(diff)
        std_diff = np.std(diff)
        label_text = f"{diff_label} (avg: {avg_diff:.2g}, std: {std_diff:.2g})"

        # Set the y-axis label with rotation and labelpad.
        twin_ax.set_ylabel(
            label_text, color=diff_color, labelpad=labelpad, rotation=270
        )

        # Match tick/label colours.
        if match_color:
            twin_ax.tick_params(axis="y", colors=diff_color)
            twin_ax.spines["right"].set_color(diff_color)
            twin_ax.yaxis.label.set_color(diff_color)

        return twin_ax

    def plot_vertical_annotations(
        self,
        position: tuple[int, int],
        positions,
        labels,
        line_color="black",
        linestyle="--",
        text_offset=(0, 5),
        **kwargs,
    ):
        """
        Draw vertical lines and add text annotations at specified positions.

        Parameters:
            position (tuple[int, int]): Tuple containing row and column indices.
            positions (list): List of x positions.
            labels (list): List of text labels for each line.
        """
        row, col = position
        ax = self.axes[row][col]
        for pos, label in zip(positions, labels):
            ax.axvline(x=pos, color=line_color, linestyle=linestyle, **kwargs)
            # Place text slightly above the top of the y-limit
            ylim = ax.get_ylim()
            ax.text(pos, ylim[1], label, ha="center", va="bottom", color=line_color)

    def save(self, filename, dpi=300, tight_layout=True):
        if tight_layout:
            self.fig.tight_layout()
        self.fig.savefig(filename, dpi=dpi, bbox_inches="tight")

    def show(self):
        plt.show()

    def plot_vertical_peak(
        self,
        position: tuple[int, int],
        target: float,
        tolerance: float = 1e-4,
        line_color: str = "black",
        linestyle: str = "--",
        label: Optional[str] = None,
        text_offset: float = 0.05,
        use_last: bool = True,
        **kwargs,
    ) -> Optional[tuple[float, float]]:
        """
        Annotate a vertical peak on the specified subplot based on previously plotted data.

        Args:
            position (tuple[int, int]): Tuple containing row and column indices.
            target (float): Target x value around which to search for the peak.
            tolerance (float, optional): Tolerance around the target value.
            line_color (str, optional): Color for the annotation.
            linestyle (str, optional): Line style for the vertical line.
            label (str, optional): Custom label for the annotation.
            text_offset (float, optional): Fraction of the y-range to offset the text.
            use_last (bool, optional): Use the most recently added plot data.
            **kwargs: Additional keyword arguments for the vertical line.

        Returns:
            Tuple containing the peak's x and y coordinates, or None if no peak is found.

        Raises:
            ValueError: If no data has been plotted on the specified subplot.
        """
        key = position
        if key not in self.plotted_data or not self.plotted_data[key]:
            raise ValueError(f"No plotted data stored for subplot {position}.")
        data_entry = (
            self.plotted_data[key][-1] if use_last else self.plotted_data[key][0]
        )
        x_data = data_entry["x"]
        y_data = data_entry["y"]

        indices = np.where(np.abs(x_data - target) <= tolerance)[0]
        if indices.size == 0:
            return None

        local_idx = np.argmax(y_data[indices])
        peak_idx = indices[local_idx]
        peak_x = x_data[peak_idx]
        peak_y = y_data[peak_idx]

        ax = self.get_axis(position)
        ax.axvline(peak_x, color=line_color, linestyle=linestyle, **kwargs)
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
        offset = text_offset * yrange
        text_str = label if label is not None else f"{peak_y:.3g}"
        ax.text(
            peak_x,
            peak_y + offset,
            text_str,
            ha="center",
            va="bottom",
            color=line_color,
        )

        return peak_x, peak_y

    def set_formatted_axis_label(
        self,
        position: tuple[int, int],
        axis,
        base_label=None,
        factor=1.0,
        offset=0.0,
        power=None,
        include_offset=True,
        unit_str=None,
        fontweight="bold",
        fontsize=16,
        color="black",
        pad=4,
    ):
        """
        Update the axis label by appending scaling information without modifying
        the tick formatting or axis limits.

        By default this method leaves the tick values untouched (using matplotlib's
        default behavior) and only updates the label text. It also allows you to specify
        a conversion factor (e.g. 1000 to convert m → mm), an offset, and/or a power-of-ten.
        Optionally you can provide a custom unit string (e.g. "mm") instead of showing the
        raw numbers.

        Parameters:
            position (tuple[int, int]): Tuple containing row and column indices.
            axis (str): Which axis to update ('x' or 'y').
            base_label (str or None): The base label text. If None, the current label is used.
            factor (float): Multiplicative factor (default 1.0).
            offset (float): Additive offset (default 0.0).
            power (int or None): If provided, indicates that the values are to be thought of as
                                multiplied by 10^(power). For instance, if power=-3, the label
                                might indicate that data are in mm rather than m.
            include_offset (bool): Whether to include offset information in the label.
            unit_str (str or None): If provided, this string (e.g. "mm") is used instead of the
                                    computed factor/power combination.
            fontweight, fontsize, color, pad: Formatting parameters for the label.
        """
        # Get the axis object.
        row, col = position
        ax = self.axes[row][col]
        # Retrieve current label if base_label is not provided.
        if base_label is None:
            if axis.lower() == "x":
                base_label = ax.get_xlabel()
            elif axis.lower() == "y":
                base_label = ax.get_ylabel()
            else:
                raise ValueError("axis must be 'x' or 'y'")

        # Build the suffix string.
        suffix = ""
        if unit_str is not None:
            suffix = unit_str
        else:
            parts = []
            if factor != 1.0:
                parts.append(f"$\\times$ {factor:g}")
            if power is not None:
                parts.append(f"$\\times$ 10^{{{power}}}")
            if include_offset and offset != 0.0:
                off_str = f"+ {offset:g}" if offset > 0 else f"- {abs(offset):g}"
                parts.append(off_str)
            suffix = " ".join(parts)

        # Only add the suffix if any parameter is different from default.
        if suffix:
            full_label = f"{base_label} [{suffix}]"
        else:
            full_label = base_label

        # Set the label.
        if axis.lower() == "x":
            ax.set_xlabel(
                full_label,
                fontweight=fontweight,
                fontsize=fontsize,
                color=color,
                labelpad=pad,
            )
        elif axis.lower() == "y":
            ax.set_ylabel(
                full_label,
                fontweight=fontweight,
                fontsize=fontsize,
                color=color,
                labelpad=pad,
            )

    # def plot_multiple_lines(
    #     self,
    #     position: tuple[int, int],
    #     lines: list[dict],
    # ):
    #     """
    #     Plot multiple lines on a specified subplot of a CustomPlot instance.

    #     Args:
    #         position (tuple[int, int]): Tuple containing row and column indices.
    #     """
    #     for line in lines:
    #         self.add_line(
    #             position,
    #             line["x"],
    #             line["y"],
    #             **{k: v for k, v in line.items() if k not in ["x", "y"]},
    #         )
    #     ax = self.get_axis(position)
    #     ax.legend()
    #     ax.grid(True, linestyle="--", alpha=0.6)


# Example usage:
if __name__ == "__main__":
    cp = CustomPlot(nrows=2, ncols=2, figsize=(12, 10), style="seaborn-v0_8-colorblind")
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    for i in range(cp.nrows):
        for j in range(cp.ncols):
            cp.add_line(
                (i, j),
                x,
                y + 0.1 * (i + j),
                label=f"Line {i},{j}",
                linestyle="-",
                marker="o",
            )

    cp.set_all_labels("x (units)", "sin(x) (units)", share_x=True, share_y=True)
    cp.remove_duplicate_ticks(share_x=True, share_y=True)
    cp.set_global_axis_limits((0, 10), (-1.5, 1.5))

    # Add secondary axis with difference on bottom-left subplot.
    cp.add_secondary_axis_with_difference(
        (1, 0), x, np.sin(x) + 0.2, np.sin(x), diff_label="Diff", diff_color="purple"
    )
    cp.set_formatted_axis_label(
        (1, 0), "x", base_label="S", factor=1, power=0, unit_str="mm"
    )
    # Use plot_vertical_peak to annotate the peak
    cp.plot_vertical_peak(
        (0, 0),
        target=5 * np.pi / 2 - 0.1,
        tolerance=0.2,
        label="Peak near 5π/2-0.1\ntol=0.2",
        line_color="red",
    )

    cp.format_legend((1, 0))
    cp.save("extended_custom_plot.png")
    cp.show()
