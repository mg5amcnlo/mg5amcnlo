#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace madspace {

std::size_t cpu_time_microsec();
std::string format_run_time(double wall_time_sec, double cpu_time_sec);
std::string format_si_prefix(double value);
std::string format_with_error(double value, double error);
std::string format_progress(double progress, int width);

class PrettyBox {
public:
    PrettyBox() = default;
    PrettyBox(
        const std::string& title,
        std::size_t rows,
        const std::vector<std::size_t>& column_sizes,
        std::size_t offset = 0,
        std::size_t box_width = 91
    );

    void print_first() const;
    void print_update() const;
    std::size_t line_count() const { return _rows + 3; }

    void set_row(std::size_t row, const std::vector<std::string>& values) {
        if (row >= _rows) {
            throw std::out_of_range("row index out of range");
        }
        for (std::size_t column = 0; auto& value : values) {
            _content.at(row * _columns + column) = value;
            ++column;
        }
    }

    void set_column(std::size_t column, const std::vector<std::string>& values) {
        if (column >= _columns) {
            throw std::out_of_range("column index out of range");
        }
        for (std::size_t row = 0; auto& value : values) {
            _content.at(row * _columns + column) = value;
            ++row;
        }
    }

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
