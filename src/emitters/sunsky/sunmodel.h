#pragma once

#include <enoki/transform.h>
#include <enoki/array_math.h>
#include <mitsuba/core/vector.h>

NAMESPACE_BEGIN(mitsuba)


#define EARTH_MEAN_RADIUS 6371.01f  // In km
#define ASTRONOMICAL_UNIT 149597890 // In km


//// Convert radians to degrees
template <typename Float> inline Float rad_to_deg(Float value) { 
    return value * (180.0f / math::Pi<Float>);
}

template <typename Float> inline Float deg_to_rad(Float value) {
    return value * (math::Pi<Float> / 180.0f);
}

template <typename Float> struct DateTimeRecord {
    int year;
    int month;
    int day;
    Float hour;
    Float minute;
    Float second;

    std::string to_string() const {
        std::ostringstream oss;
        oss << "DateTimeRecord[year = " << year
            << ", month= " << month
            << ", day = " << day
            << ", hour = " << hour
            << ", minute = " << minute
            << ", second = " << second << "]";
        return oss.str();
    }
};

template <typename Float> struct LocationRecord {
    Float longitude;
    Float latitude;
    Float timezone;

    std::string to_string() const {
        std::ostringstream oss;
        oss << "LocationRecord[latitude = " << latitude
            << ", longitude = " << longitude
            << ", timezone = " << timezone << "]";
        return oss.str();
    }
};

template <typename Float> struct SphericalCoordinates {
    Float elevation;
    Float azimuth;

    inline SphericalCoordinates() { }

    inline SphericalCoordinates(Float elevation, Float azimuth)
        : elevation(elevation), azimuth(azimuth) { }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SphericalCoordinates[elevation = " << rad_to_deg(elevation)
            << ", azimuth = " << rad_to_deg(azimuth) << "]";
        return oss.str();
    }
};

// template <typename Float> Vector toSphere(const SphericalCoordinates<Float> coords) {
//     Float sinTheta, cosTheta, sinPhi, cosPhi;

//     math::sincos(coords.elevation, &sinTheta, &cosTheta);
//     math::sincos(coords.azimuth, &sinPhi, &cosPhi);

//     return Vector(sinPhi*sinTheta, cosTheta, -cosPhi*sinTheta);
// }

/**
 * \brief Compute the spherical coordinates of the vector \c d
 */
template <typename Float>
SphericalCoordinates<Float> from_sphere(const Vector<Float, 3> &d) {
    Float azimuth = atan2(d.x(), -d.z());
    Float elevation = safe_acos(d.y());
    
    if (azimuth < 0)
        azimuth += 2 * math::Pi<Float>;

    return SphericalCoordinates(elevation, azimuth);
}

/**
 * \brief S-shaped smoothly varying interpolation between two values
 */
template <typename Float>
Float smooth_step(Float min, Float max, Float value) {
    Float v = clamp((value - min) / (max - min), (Float) 0, (Float) 1);
    return v * v * (-2 * v  + 3);
}

/**
 * \brief Compute the elevation and azimuth of the sun as seen by an observer
 * at \c location at the date and time specified in \c date_time.
 *
 * Based on "Computing the Solar Vector" by Manuel Blanco-Muriel,
 * Diego C. Alarcon-Padilla, Teodoro Lopez-Moratalla, and Martin Lara-Coira,
 * in "Solar energy", vol 27, number 5, 2001 by Pergamon Press.
 */
template <typename Float>
SphericalCoordinates<Float> compute_sun_coordinates(const DateTimeRecord<Float> &date_time, const LocationRecord<Float> &location) {
    // Main variables
    Float elapsed_julian_days, dec_hours;
    Float ecliptic_longitude, ecliptic_obliquity;
    Float right_ascension, declination;
    Float elevation, azimuth;

    // Auxiliary variables
    Float d_y;
    Float d_x;

    /* Calculate difference in days between the current Julian Day
       and JD 2451545.0, which is noon 1 January 2000 Universal Time */
    {
        // Calculate time of the day in UT decimal hours
        dec_hours = date_time.hour - location.timezone +
            (date_time.minute + date_time.second / 60.0f ) / 60.0f;

        // Calculate current Julian Day
        int li_aux1 = (date_time.month-14) / 12;
        int li_aux2 = (1461*(date_time.year + 4800 + li_aux1)) / 4
            + (367 * (date_time.month - 2 - 12 * li_aux1)) / 12
            - (3 * ((date_time.year + 4900 + li_aux1) / 100)) / 4
            + date_time.day - 32075;
        Float d_julian_date = li_aux2 - 0.5f + dec_hours / 24.0f;

        // Calculate difference between current Julian Day and JD 2451545.0
        elapsed_julian_days = d_julian_date - 2451545.0f;
    }

    /* Calculate ecliptic coordinates (ecliptic longitude and obliquity of the
       ecliptic in radians but without limiting the angle to be less than 2*Pi
       (i.e., the result may be greater than 2*Pi) */
    {
        Float omega = 2.1429f - 0.0010394594f * elapsed_julian_days;
        Float mean_longitude = 4.8950630f + 0.017202791698f * elapsed_julian_days; // Radians
        Float anomaly = 6.2400600f + 0.0172019699f * elapsed_julian_days;

        ecliptic_longitude = mean_longitude + 0.03341607f * sin(anomaly)
            + 0.00034894f * sin(2*anomaly) - 0.0001134f
            - 0.0000203f * sin(omega);

        ecliptic_obliquity = 0.4090928f - 6.2140e-9f * elapsed_julian_days
            + 0.0000396f * cos(omega);
    }

    /* Calculate celestial coordinates ( right ascension and declination ) in radians
       but without limiting the angle to be less than 2*Pi (i.e., the result may be
       greater than 2*Pi) */
    {
        Float sinecliptic_longitude = sin(ecliptic_longitude);
        d_y = cos(ecliptic_obliquity) * sinecliptic_longitude;
        d_x = cos(ecliptic_longitude);
        right_ascension = atan2(d_y, d_x);
        if (right_ascension < 0.0f)
            right_ascension += 2*math::Pi<Float>;
        declination = asin(sin(ecliptic_obliquity) * sinecliptic_longitude);
    }

    // Calculate local coordinates (azimuth and zenith angle) in degrees
    {
        Float greenwich_mean_sidereal_time = 6.6974243242f
            + 0.0657098283f * elapsed_julian_days + dec_hours;

        Float local_mean_sidereal_time = deg_to_rad((greenwich_mean_sidereal_time * 15
            + location.longitude));

        Float latitude_in_radians = deg_to_rad(location.latitude);
        Float cos_latitude = cos(latitude_in_radians);
        Float sin_latitude = sin(latitude_in_radians);

        Float hour_angle = local_mean_sidereal_time - right_ascension;
        Float coshour_angle = cos(hour_angle);

        elevation = acos(cos_latitude * coshour_angle
            * cos(declination) + sin(declination) * sin_latitude);

        d_y = -sin(hour_angle);
        d_x = tan(declination) * cos_latitude - sin_latitude * coshour_angle;

        azimuth = atan2(d_y, d_x);
        if (azimuth < 0.0f)
            azimuth += 2*math::Pi<Float>;

        // Parallax Correction
        elevation += (EARTH_MEAN_RADIUS / ASTRONOMICAL_UNIT) * (Float)sin((double)elevation);
    }

    return SphericalCoordinates((Float) elevation, (Float) azimuth);
}

template <typename Float, typename Point>
SphericalCoordinates<Float> compute_sun_coordinates(
    const Vector<Float, 3>& sun_dir, const Transform<Point> &world_to_luminaire) {
        Float x = world_to_luminaire.matrix[0][0] * sun_dir[0] + world_to_luminaire.matrix[0][1] * sun_dir[1]
                + world_to_luminaire.matrix[0][2] * sun_dir[2];
        Float y = world_to_luminaire.matrix[1][0] * sun_dir[0] + world_to_luminaire.matrix[1][1] * sun_dir[1]
                + world_to_luminaire.matrix[1][2] * sun_dir[2];
        Float z = world_to_luminaire.matrix[2][0] * sun_dir[0] + world_to_luminaire.matrix[2][1] * sun_dir[1]
                + world_to_luminaire.matrix[2][2] * sun_dir[2];

        return from_sphere(normalize(Vector<Float, 3>(x, y, z)));
}

template <typename Float>
SphericalCoordinates<Float> compute_sun_coordinates(const Properties &props) {
    if (props.has_property("sun_direction")) {
        if (props.has_property("latitude") || props.has_property("longitude")
            || props.has_property("timezone") || props.has_property("day")
            || props.has_property("time"))
            Log(Error, "Both the 'sun_direction' parameter and time/location "
                    "information were provided -- only one of them can be specified at a time!");
        
        return compute_sun_coordinates<Float>(props.vector3f("sun_direction"),
            props.animated_transform("to_world", Transform<Point<Float, 4>>())->eval(0).inverse());
    } else {
        LocationRecord<Float> location;
        DateTimeRecord<Float> date_time;

        location.latitude = props.float_("latitude", 35.6894f);
        location.longitude = props.float_("longitude", 139.6917f);
        location.timezone = props.float_("timezone", 9);
        date_time.year = props.int_("year", 2010);
        date_time.month = props.int_("month", 7);
        date_time.day = props.int_("day", 10);
        date_time.hour = props.float_("hour", 15.0f);
        date_time.minute = props.float_("minute", 0.0f);
        date_time.second = props.float_("second", 0.0f);

        SphericalCoordinates coords = compute_sun_coordinates<Float>(date_time, location);

        return coords;
    }
}

NAMESPACE_END(mitsuba)