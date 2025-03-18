module Output

using Printf: @sprintf

using ..ComparatorNetworks: ComparatorNetwork

################################################################# TEXTUAL OUTPUT


export hexfloat


function hexfloat(x::Float32)
    # TODO: Subnormal numbers should be printed with leading zeros.
    @assert isfinite(x)
    @assert precision(x) == 24
    s = @sprintf("%+.6A", x)
    @assert length(s) >= 14
    @assert (s[1] == '+') | (s[1] == '-')
    @assert s[2] == '0'
    @assert s[3] == 'X'
    @assert (s[4] == '0') | (s[4] == '1')
    @assert s[5] == '.'
    @assert all(('0' <= s[i] <= '9') | ('A' <= s[i] <= 'F') for i = 6:11)
    @assert s[12] == 'P'
    @assert (s[13] == '+') | (s[13] == '-')
    @assert all('0' <= s[i] <= '9' for i = 14:length(s))
    return @sprintf("%c0x%c.%sp%+04d",
        s[1], s[4], s[6:11], parse(Int, s[13:end]))
end


function hexfloat(x::Float64)
    # TODO: Subnormal numbers should be printed with leading zeros.
    @assert isfinite(x)
    @assert precision(x) == 53
    s = @sprintf("%+.13A", x)
    @assert length(s) >= 21
    @assert (s[1] == '+') | (s[1] == '-')
    @assert s[2] == '0'
    @assert s[3] == 'X'
    @assert (s[4] == '0') | (s[4] == '1')
    @assert s[5] == '.'
    @assert all(('0' <= s[i] <= '9') | ('A' <= s[i] <= 'F') for i = 6:18)
    @assert s[19] == 'P'
    @assert (s[20] == '+') | (s[20] == '-')
    @assert all('0' <= s[i] <= '9' for i = 21:length(s))
    return @sprintf("%c0x%c.%sp%+05d",
        s[1], s[4], s[6:18], parse(Int, s[20:end]))
end


hexfloat(t::Tuple) = '(' * join(hexfloat.(t), ", ") * ')'


############################################################### GRAPHICAL OUTPUT


export svg_string


svg_string(x::Float64) = rstrip(rstrip(@sprintf("%.15f", x), '0'), '.')


function svg_string(
    x_min::Float64, x_max::Float64,
    y_min::Float64, y_max::Float64,
)
    width_string = svg_string(x_max - x_min)
    height_string = svg_string(y_max - y_min)
    return @sprintf(
        """<svg xmlns="%s" viewBox="%s %s %s %s" width="%s" height="%s">""",
        "http://www.w3.org/2000/svg", svg_string(x_min), svg_string(y_min),
        width_string, height_string, width_string, height_string)
end


struct _SVGCircle
    cx::Float64
    cy::Float64
    r::Float64
end


@inline _min_x(circle::_SVGCircle) = circle.cx - circle.r
@inline _min_y(circle::_SVGCircle) = circle.cy - circle.r
@inline _max_x(circle::_SVGCircle) = circle.cx + circle.r
@inline _max_y(circle::_SVGCircle) = circle.cy + circle.r


svg_string(circle::_SVGCircle, color::String, width::Float64) = @sprintf(
    """<circle cx="%s" cy="%s" r="%s" stroke="%s" stroke-width="%s" fill="none" />""",
    svg_string(circle.cx), svg_string(circle.cy), svg_string(circle.r),
    color, svg_string(width))


struct _SVGLine
    x1::Float64
    y1::Float64
    x2::Float64
    y2::Float64
end


@inline _min_x(line::_SVGLine) = min(line.x1, line.x2)
@inline _min_y(line::_SVGLine) = min(line.y1, line.y2)
@inline _max_x(line::_SVGLine) = max(line.x1, line.x2)
@inline _max_y(line::_SVGLine) = max(line.y1, line.y2)


svg_string(line::_SVGLine, color::String, width::Float64) = @sprintf(
    """<line x1="%s" y1="%s" x2="%s" y2="%s" stroke="%s" stroke-width="%s" />""",
    svg_string(line.x1), svg_string(line.y1),
    svg_string(line.x2), svg_string(line.y2),
    color, svg_string(width))


@enum _ColumnOccupation begin
    _NONE
    _WEAK
    _STRONG
end


function svg_string(
    network::ComparatorNetwork{N};
    line_height::Float64=32.0,
    circle_radius::Float64=8.0,
    minor_width::Float64=NaN,
    major_width::Float64=NaN,
    padding_left::Float64=NaN,
    padding_right::Float64=NaN,
    padding_top::Float64=NaN,
    padding_bottom::Float64=NaN,
) where {N}

    # `line_height` and `circle_radius` must be supplied by the user.
    @assert isfinite(line_height)
    @assert isfinite(circle_radius)

    # The remaining parameters are optional with the following default values.
    isfinite(minor_width) || (minor_width = 2.0 * circle_radius)
    isfinite(major_width) || (major_width = 5.0 * circle_radius)
    isfinite(padding_left) || (padding_left = 3.0 * circle_radius)
    isfinite(padding_right) || (padding_right = 3.0 * circle_radius)
    isfinite(padding_top) || (padding_top = 2.0 * circle_radius)
    isfinite(padding_bottom) || (padding_bottom = 2.0 * circle_radius)

    lines = _SVGLine[]
    circles = _SVGCircle[]
    occupied = _ColumnOccupation[_NONE for _ = 1:N]
    x = 0.0

    for (i, j) in network.comparators
        @assert 1 <= i < j <= N
        if (occupied[i] == _STRONG) || (occupied[j] == _STRONG)
            occupied .= _NONE
            x += major_width
        elseif any(occupied[k] != _NONE for k = i:j)
            x += minor_width
        end
        for k = i:j
            if occupied[k] == _NONE
                occupied[k] = _WEAK
            end
        end
        occupied[i] = _STRONG
        occupied[j] = _STRONG
        yi = i * line_height
        yj = j * line_height
        push!(lines, _SVGLine(x, yi - circle_radius, x, yj + circle_radius))
        push!(circles, _SVGCircle(x, yi, circle_radius))
        push!(circles, _SVGCircle(x, yj, circle_radius))
    end

    for i = 1:N
        push!(lines, _SVGLine(
            -padding_left, i * line_height,
            x + padding_right, i * line_height))
    end

    buf = IOBuffer()
    print(buf, svg_string(-padding_left, x + padding_right,
        line_height - padding_top, N * line_height + padding_bottom))

    for line in lines
        print(buf, svg_string(line, "white", 3.0))
    end
    for line in lines
        print(buf, svg_string(line, "black", 1.0))
    end
    for circle in circles
        print(buf, svg_string(circle, "white", 3.0))
    end
    for circle in circles
        print(buf, svg_string(circle, "black", 1.0))
    end

    print(buf, "</svg>")
    return String(take!(buf))
end


function Base.show(
    io::IO,
    ::MIME"text/html",
    network::ComparatorNetwork{N},
) where {N}
    println(io, svg_string(network))
end


################################################################################

end # module Output
