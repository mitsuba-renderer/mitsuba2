#pragma once

#include <enoki/transform.h>
#include <enoki/array_math.h>
#include <mitsuba/core/vector.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/distr_1d.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/interaction.h>

NAMESPACE_BEGIN(mitsuba)


#define EARTH_MEAN_RADIUS 6371.01  // In km
#define ASTRONOMICAL_UNIT 149597890 // In km


template <typename Float> inline Float rad_to_deg(Float value) { 
    return value * (180.0f * math::InvPi<Float>);
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

template <typename Float>
Vector<Float, 3> to_sphere(const SphericalCoordinates<Float> coords) {
    Float sin_theta = sin(coords.elevation);
    Float cos_theta = cos(coords.elevation);
    Float sin_phi = sin(coords.azimuth);
    Float cos_phi = cos(coords.azimuth);

    return Vector<Float, 3>(sin_phi * sin_theta, cos_theta, -cos_phi * sin_theta);
}

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
    return v * v * (-2 * v + 3);
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
    double elapsed_julian_days, dec_hours;
    double ecliptic_longitude, ecliptic_obliquity;
    double right_ascension, declination;
    double elevation, azimuth;

    // Auxiliary variables
    double d_y;
    double d_x;

    /* Calculate difference in days between the current Julian Day
       and JD 2451545.0, which is noon 1 January 2000 Universal Time */
    {
        // Calculate time of the day in UT decimal hours
        dec_hours = (double) date_time.hour - (double) location.timezone +
            ((double) date_time.minute + (double) date_time.second / 60.0 ) / 60.0;

        // Calculate current Julian Day
        int li_aux1 = (date_time.month-14) / 12;
        int li_aux2 = (1461*(date_time.year + 4800 + li_aux1)) / 4
            + (367 * (date_time.month - 2 - 12 * li_aux1)) / 12
            - (3 * ((date_time.year + 4900 + li_aux1) / 100)) / 4
            + date_time.day - 32075;
        double d_julian_date = (double) li_aux2 - 0.5 + dec_hours / 24.0;

        // Calculate difference between current Julian Day and JD 2451545.0
        elapsed_julian_days = d_julian_date - 2451545.0;
    }

    /* Calculate ecliptic coordinates (ecliptic longitude and obliquity of the
       ecliptic in radians but without limiting the angle to be less than 2*Pi
       (i.e., the result may be greater than 2*Pi) */
    {
        double omega = 2.1429 - 0.0010394594 * elapsed_julian_days;
        double mean_longitude = 4.8950630 + 0.017202791698 * elapsed_julian_days; // Radians
        double anomaly = 6.2400600 + 0.0172019699 * elapsed_julian_days;

        ecliptic_longitude = mean_longitude + 0.03341607 * sin(anomaly)
            + 0.00034894 * sin(2*anomaly) - 0.0001134
            - 0.0000203 * sin(omega);

        ecliptic_obliquity = 0.4090928 - 6.2140e-9 * elapsed_julian_days
            + 0.0000396 * cos(omega);
    }

    /* Calculate celestial coordinates ( right ascension and declination ) in radians
       but without limiting the angle to be less than 2*Pi (i.e., the result may be
       greater than 2*Pi) */
    {
        double sinecliptic_longitude = sin(ecliptic_longitude);
        d_y = cos(ecliptic_obliquity) * sinecliptic_longitude;
        d_x = cos(ecliptic_longitude);
        right_ascension = atan2(d_y, d_x);
        if (right_ascension < 0.0)
            right_ascension += 2*math::Pi<double>;
        declination = asin(sin(ecliptic_obliquity) * sinecliptic_longitude);
    }

    // Calculate local coordinates (azimuth and zenith angle) in degrees
    {
        double greenwich_mean_sidereal_time = 6.6974243242
            + 0.0657098283 * elapsed_julian_days + dec_hours;

        double local_mean_sidereal_time = (double) deg_to_rad((Float) (greenwich_mean_sidereal_time * 15
            + (double) location.longitude));

        double latitude_in_radians = (double) deg_to_rad(location.latitude);
        double cos_latitude = cos(latitude_in_radians);
        double sin_latitude = sin(latitude_in_radians);

        double hour_angle = local_mean_sidereal_time - right_ascension;
        double coshour_angle = cos(hour_angle);

        elevation = acos(cos_latitude * coshour_angle
            * cos(declination) + sin(declination) * sin_latitude);

        d_y = -sin(hour_angle);
        d_x = tan(declination) * cos_latitude - sin_latitude * coshour_angle;

        azimuth = atan2(d_y, d_x);
        if (azimuth < 0.0)
            azimuth += 2*math::Pi<double>;

        // Parallax Correction
        elevation += (EARTH_MEAN_RADIUS / ASTRONOMICAL_UNIT) * sin(elevation);
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

/* The following is from the implementation of "A Practical Analytic Model for
   Daylight" by A.J. Preetham, Peter Shirley, and Brian Smits */

/* All data lifted from MI. Units are either [] or cm^-1. refer when in doubt MI */

// k_o Spectrum table from pg 127, MI.
template <typename Float>
Float k_o_wavelengths[64] = {
    300, 305, 310, 315, 320, 325, 330, 335, 340, 345,
    350, 355, 445, 450, 455, 460, 465, 470, 475, 480,
    485, 490, 495, 500, 505, 510, 515, 520, 525, 530,
    535, 540, 545, 550, 555, 560, 565, 570, 575, 580,
    585, 590, 595, 600, 605, 610, 620, 630, 640, 650,
    660, 670, 680, 690, 700, 710, 720, 730, 740, 750,
    760, 770, 780, 790
};

template <typename Float>
Float k_o_amplitudes[65] = {
    10.0, 4.8, 2.7, 1.35, .8, .380, .160, .075, .04, .019, .007,
    .0, .003, .003, .004, .006, .008, .009, .012, .014, .017,
    .021, .025, .03, .035, .04, .045, .048, .057, .063, .07,
    .075, .08, .085, .095, .103, .110, .12, .122, .12, .118,
    .115, .12, .125, .130, .12, .105, .09, .079, .067, .057,
    .048, .036, .028, .023, .018, .014, .011, .010, .009,
    .007, .004, .0, .0
};

// k_g Spectrum table from pg 130, MI.
template <typename Float>
Float k_g_wavelengths[4] = {
    759, 760, 770, 771
};

template <typename Float>
Float k_g_amplitudes[4] = {
    0, 3.0, 0.210, 0
};

// k_wa Spectrum table from pg 130, MI.
template <typename Float>
Float k_wa_wavelengths[13] = {
    689, 690, 700, 710, 720,
    730, 740, 750, 760, 770,
    780, 790, 800
};

template <typename Float>
Float k_wa_amplitudes[13] = {
    0, 0.160e-1, 0.240e-1, 0.125e-1,
    0.100e+1, 0.870, 0.610e-1, 0.100e-2,
    0.100e-4, 0.100e-4, 0.600e-3,
    0.175e-1, 0.360e-1
};

/* Wavelengths corresponding to the table below */
template <typename Float>
Float sol_wavelengths[38] = {
    380, 390, 400, 410, 420, 430, 440, 450,
    460, 470, 480, 490, 500, 510, 520, 530,
    540, 550, 560, 570, 580, 590, 600, 610,
    620, 630, 640, 650, 660, 670, 680, 690,
    700, 710, 720, 730, 740, 750
};

/* Solar amplitude in watts / (m^2 * nm * sr) */
template <typename Float>
Float sol_amplitudes[38] = {
    16559.0, 16233.7, 21127.5, 25888.2, 25829.1,
    24232.3, 26760.5, 29658.3, 30545.4, 30057.5,
    30663.7, 28830.4, 28712.1, 27825.0, 27100.6,
    27233.6, 26361.3, 25503.8, 25060.2, 25311.6,
    25355.9, 25134.2, 24631.5, 24173.2, 23685.3,
    23212.1, 22827.7, 22339.8, 21970.2, 21526.7,
    21097.9, 20728.3, 20240.4, 19870.8, 19427.2,
    19072.4, 18628.9, 18259.2
};

template <typename Float, typename Spectrum>
ref<Texture<Float, Spectrum>> compute_sun_radiance(Float theta, Float turbidity, Float factor) {
    using Texture = Texture<Float, Spectrum>;
    using IrregularContinuousDistribution = IrregularContinuousDistribution<Float>;
    // TODO: use ScalarFloat

    IrregularContinuousDistribution k_o_curve(k_o_wavelengths<Float>, k_o_amplitudes<Float>, 64);
    IrregularContinuousDistribution k_g_curve(k_g_wavelengths<Float>, k_g_amplitudes<Float>, 4);
    IrregularContinuousDistribution k_wa_curve(k_wa_wavelengths<Float>, k_wa_amplitudes<Float>, 13);
    IrregularContinuousDistribution sol_curve(sol_wavelengths<Float>, sol_amplitudes<Float>, 38);
    
    std::vector<Float> data(91), wavelengths(91);  // (800 - 350) / 5  + 1

    Float beta = 0.04608365822050f * turbidity - 0.04586025928522f;

    // Relative Optical Mass
    Float m = 1.0f / ((Float) cos(theta) + 0.15f *
        pow(93.885f - theta * math::InvPi<Float> * 180.0f, (Float) -1.253f));
    
    Float lambda;
    int i = 0;
    for (i = 0, lambda = 350; i < 91; i++, lambda += 5) {

        // Rayleigh Scattering
        // Results agree with the graph (pg 115, MI) */
        Float tau_r = exp(-m * 0.008735f * pow(lambda / 1000.0f, (Float) -4.08));

        // Aerosol (water + dust) attenuation
        // beta - amount of aerosols present
        // alpha - ratio of small to large particle sizes. (0:4,usually 1.3)
        // Results agree with the graph (pg 121, MI)
        const Float alpha = 1.3f;
        Float tau_a = exp(-m * beta * pow(lambda / 1000.0f, -alpha));  // lambda should be in um

        // Attenuation due to ozone absorption
        // lOzone - amount of ozone in cm(NTP)
        // Results agree with the graph (pg 128, MI)
        const Float l_ozone = .35f;
        Float tau_o = exp(-m * k_o_curve.eval_pdf(lambda) * l_ozone);

        // Attenuation due to mixed gases absorption
        // Results agree with the graph (pg 131, MI)
        Float tau_g = exp(-1.41f * k_g_curve.eval_pdf(lambda) * m / pow(1 + 118.93f
            * k_g_curve.eval_pdf(lambda) * m, (Float) 0.45f));

        // Attenuation due to water vapor absorbtion
        // w - precipitable water vapor in centimeters (standard = 2)
        // Results agree with the graph (pg 132, MI)
        const Float w = 2.0f;
        Float tau_wa = exp(-0.2385f * k_wa_curve.eval_pdf(lambda) * w * m /
                pow(1 + 20.07f * k_wa_curve.eval_pdf(lambda) * w * m, (Float) 0.45f));

        data[i] = sol_curve.eval_pdf(lambda) * tau_r * tau_a * tau_o * tau_g * tau_wa * factor;
        wavelengths[i] = lambda;
    }
        
    if constexpr (is_rgb_v<Spectrum>) {
        Color<Float, 3> color = spectrum_to_rgb<Float>(wavelengths, data, false) * MTS_CIE_Y_NORMALIZATION;

        Properties props("srgb");
        props.set_color("color", color);
        props.set_bool("unbounded", true);
        ref<Texture> discretized = PluginManager::instance()->create_object<Texture>(props);
        
        return discretized;
    } else {
        Properties props("regular");
        props.set_pointer("wavelengths", wavelengths.data());
        props.set_pointer("values", data.data());
        props.set_long("size", wavelengths.size());
        ref<Texture> interpolated = PluginManager::instance()->create_object<Texture>(props);
        return interpolated;
    }
}

NAMESPACE_END(mitsuba)
