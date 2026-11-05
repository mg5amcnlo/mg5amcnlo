#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace madspace {

/// CPU time consumed by the current process, in microseconds.
std::size_t cpu_time_microsec();
/// Human-readable `wall_time_sec` and `cpu_time_sec`, formatted as
/// `"HH:MM:SS.cc wall, HH:MM:SS.cc cpu"`.
std::string format_run_time(double wall_time_sec, double cpu_time_sec);
/// `value`, rounded to an integer, with an SI magnitude suffix (`k`, `M`, ...)
/// and three significant digits.
std::string format_si_prefix(double value);
/// `value` with its uncertainty `error`, e.g. `"1.234(45)"`. If `error` is not
/// smaller than `value`, the form `"value ± error"` is used.
std::string format_with_error(double value, double error);
/// Progress bar of `width` Unicode block characters, `progress` in `[0, 1]`.
std::string format_progress(double progress, int width);

/**
 * Fixed-layout text table redrawn in place on the terminal.
 *
 * Used for the live progress display during event generation (`Verbosity::pretty`).
 * @ref print_first draws the initial box; @ref print_update moves
 * the cursor back up and repaints the same lines, so the box appears to
 * update rather than scroll.
 */
class PrettyBox {
public:
    /// Empty box; must not be printed.
    PrettyBox() = default;
    /**
     * @param title         Text shown in the top border.
     * @param rows          Number of content rows.
     * @param column_sizes  Width of each column, in characters.
     * @param offset        Extra lines between the end of the box and the cursor
     *                      when it is redrawn.
     * @param box_width     Minimum box width; the last column is widened to it.
     */
    PrettyBox(
        const std::string& title,
        std::size_t rows,
        const std::vector<std::size_t>& column_sizes,
        std::size_t offset = 0,
        std::size_t box_width = 91
    );

    /// Draw the box for the first time.
    void print_first() const;
    /// Redraw the box in place with the current cell contents.
    void print_update() const;
    /// Number of terminal lines the box occupies, including its borders.
    std::size_t line_count() const { return _rows + 3; }

    /// Set every cell of @p row.
    void set_row(std::size_t row, const std::vector<std::string>& values) {
        if (row >= _rows) {
            throw std::out_of_range("row index out of range");
        }
        for (std::size_t column = 0; auto& value : values) {
            _content.at(row * _columns + column) = value;
            ++column;
        }
    }

    /// Set every cell of @p column.
    void set_column(std::size_t column, const std::vector<std::string>& values) {
        if (column >= _columns) {
            throw std::out_of_range("column index out of range");
        }
        for (std::size_t row = 0; auto& value : values) {
            _content.at(row * _columns + column) = value;
            ++row;
        }
    }

    /// Set a single cell.
    void set_cell(std::size_t row, std::size_t column, std::string value) {
        if (row >= _rows) {
            throw std::out_of_range("row index out of range");
        }
        if (column >= _columns) {
            throw std::out_of_range("column index out of range");
        }
        _content.at(row * _columns + column) = value;
    }

private:
    std::string _header;
    std::string _footer;
    std::size_t _rows;
    std::size_t _columns;
    std::size_t _offset;
    std::vector<std::size_t> _column_ends;
    std::vector<std::string> _content;
};

} // namespace madspace
