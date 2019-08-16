-- Author: Kate Storey-Fisher @kstoreyf

SELECT

    -- Basic information
    f1.object_id, f1.ra, f1.dec, f1.tract, f1.patch, f1.parent_id,

    -- Galactic extinction
    f1.a_g, f1.a_r, f1.a_i, f1.a_z, f1.a_y,

    -- Photometry

    --- cmodel
    ---- Total
    f1.g_cmodel_mag, f1.r_cmodel_mag, f1.i_cmodel_mag, f1.z_cmodel_mag, f1.y_cmodel_mag,
    f1.g_cmodel_magsigma, f1.r_cmodel_magsigma, f1.i_cmodel_magsigma, f1.z_cmodel_magsigma, f1.y_cmodel_magsigma,

    ---- fracDev
    f1.g_cmodel_fracdev, f1.r_cmodel_fracdev, f1.i_cmodel_fracdev, f1.z_cmodel_fracdev, f1.y_cmodel_fracdev,

    ---- flag
    f1.g_cmodel_flag, f1.r_cmodel_flag, f1.i_cmodel_flag, f1.z_cmodel_flag, f1.y_cmodel_flag,

    --- PSF
    f2.g_psfflux_mag, f2.r_psfflux_mag, f2.i_psfflux_mag, f2.z_psfflux_mag, f2.y_psfflux_mag,
    f2.g_psfflux_magsigma, f2.r_psfflux_magsigma, f2.i_psfflux_magsigma, f2.z_psfflux_magsigma, f2.y_psfflux_magsigma,

    ---- flag
    f2.g_psfflux_flag, f2.r_psfflux_flag, f2.i_psfflux_flag, f2.z_psfflux_flag, f2.y_psfflux_flag,

    -- Flags
    ---- pixel edge
    f1.g_pixelflags_edge, f1.r_pixelflags_edge, f1.i_pixelflags_edge, f1.z_pixelflags_edge, f1.y_pixelflags_edge,

    ---- pixel interpolated
    f1.g_pixelflags_interpolated, f1.r_pixelflags_interpolated, f1.i_pixelflags_interpolated, f1.z_pixelflags_interpolated,
    f1.y_pixelflags_interpolated,

    ---- pixel saturated
    f1.g_pixelflags_saturated, f1.r_pixelflags_saturated, f1.i_pixelflags_saturated, f1.z_pixelflags_saturated,
    f1.y_pixelflags_saturated,

    ---- pixel cr
    f1.g_pixelflags_cr, f1.r_pixelflags_cr, f1.i_pixelflags_cr, f1.z_pixelflags_cr, f1.y_pixelflags_cr,

    ---- pixel clipped
    f1.g_pixelflags_clipped, f1.r_pixelflags_clipped, f1.i_pixelflags_clipped, f1.z_pixelflags_clipped,
    f1.y_pixelflags_clipped,

    ---- pixel reject
    f1.g_pixelflags_rejected, f1.r_pixelflags_rejected, f1.i_pixelflags_rejected, f1.z_pixelflags_rejected,
    f1.y_pixelflags_rejected,

    ---- pixel inexact psf
    f1.g_pixelflags_inexact_psf, f1.r_pixelflags_inexact_psf, f1.i_pixelflags_inexact_psf,
    f1.z_pixelflags_inexact_psf, f1.y_pixelflags_inexact_psf,

    ---- pixel interpolated center
    f1.g_pixelflags_interpolatedcenter, f1.r_pixelflags_interpolatedcenter, f1.i_pixelflags_interpolatedcenter,
    f1.z_pixelflags_interpolatedcenter, f1.y_pixelflags_interpolatedcenter,

    ---- pixel saturated center
    f1.g_pixelflags_saturatedcenter, f1.r_pixelflags_saturatedcenter, f1.i_pixelflags_saturatedcenter, f1.z_pixelflags_saturatedcenter,
    f1.y_pixelflags_saturatedcenter,

    ---- pixel cr center
    f1.g_pixelflags_crcenter, f1.r_pixelflags_crcenter, f1.i_pixelflags_crcenter, f1.z_pixelflags_crcenter, f1.y_pixelflags_crcenter,

    ---- pixel clipped center
    f1.g_pixelflags_clippedcenter, f1.r_pixelflags_clippedcenter, f1.i_pixelflags_clippedcenter, f1.z_pixelflags_clippedcenter,
    f1.y_pixelflags_clippedcenter,

    ---- pixel reject center
    f1.g_pixelflags_rejectedcenter, f1.r_pixelflags_rejectedcenter, f1.i_pixelflags_rejectedcenter, f1.z_pixelflags_rejectedcenter,
    f1.y_pixelflags_rejectedcenter,

    ---- pixel inexact psf center
    f1.g_pixelflags_inexact_psfcenter, f1.r_pixelflags_inexact_psfcenter, f1.i_pixelflags_inexact_psfcenter,
    f1.z_pixelflags_inexact_psfcenter, f1.y_pixelflags_inexact_psfcenter,

    ---- pixel bright object
    f1.g_pixelflags_bright_object, f1.r_pixelflags_bright_object, f1.i_pixelflags_bright_object,
    f1.z_pixelflags_bright_object, f1.y_pixelflags_bright_object,

    ---- pixel bright object center
    f1.g_pixelflags_bright_objectcenter, f1.r_pixelflags_bright_objectcenter, f1.i_pixelflags_bright_objectcenter,
    f1.z_pixelflags_bright_objectcenter, f1.y_pixelflags_bright_objectcenter,

    -- Meta information
    ---- input count
    f1.g_inputcount_value, f1.r_inputcount_value, f1.i_inputcount_value, f1.z_inputcount_value, f1.y_inputcount_value,

    ---- extendedness
    f1.g_extendedness_value, f1.r_extendedness_value, f1.i_extendedness_value, f1.z_extendedness_value, f1.y_extendedness_value,
    f1.g_extendedness_flag, f1.r_extendedness_flag, f1.i_extendedness_flag, f1.z_extendedness_flag, f1.y_extendedness_flag,

    ---- background
    f1.g_localbackground_flux, f1.r_localbackground_flux, f1.i_localbackground_flux,
    f1.z_localbackground_flux, f1.y_localbackground_flux

FROM
    pdr2_wide.forced AS f1
    LEFT JOIN pdr2_wide.forced2 AS f2 USING (object_id)

WHERE

    -- Make sure we only select the primary target
    f1.isprimary = True
AND f1.nchild = 0

    -- HSC Wide is separated into 7 sub-regions from W01 to W07
    -- You can only select objects from one region using :
-- AND s18a_wide.search_w02(object_id)

    -- Rough FDFC cuts
AND f1.g_inputcount_value >= 3
AND f1.r_inputcount_value >= 3
AND f1.i_inputcount_value >= 3
AND f1.z_inputcount_value >= 3
AND f1.y_inputcount_value >= 3

    -- If you want to select star or galaxy
    -- Extended objects = 1; Point source = 0
-- AND f1.i_extendedness_value = 1
-- AND f1.r_extendedness_value = 1

AND NOT f1.g_pixelflags_bright_objectcenter
AND NOT f1.r_pixelflags_bright_objectcenter
AND NOT f1.i_pixelflags_bright_objectcenter
AND NOT f1.z_pixelflags_bright_objectcenter
AND NOT f1.y_pixelflags_bright_objectcenter

AND NOT f1.g_pixelflags_bright_object
AND NOT f1.r_pixelflags_bright_object
AND NOT f1.i_pixelflags_bright_object
AND NOT f1.z_pixelflags_bright_object
AND NOT f1.y_pixelflags_bright_object

AND NOT f1.g_pixelflags_edge
AND NOT f1.r_pixelflags_edge
AND NOT f1.i_pixelflags_edge
AND NOT f1.z_pixelflags_edge
AND NOT f1.y_pixelflags_edge

AND NOT f1.g_pixelflags_saturatedcenter
AND NOT f1.r_pixelflags_saturatedcenter
AND NOT f1.i_pixelflags_saturatedcenter
AND NOT f1.z_pixelflags_saturatedcenter
AND NOT f1.y_pixelflags_saturatedcenter

AND NOT f1.g_cmodel_flag
AND NOT f1.r_cmodel_flag
AND NOT f1.i_cmodel_flag
AND NOT f1.z_cmodel_flag
AND NOT f1.y_cmodel_flag

    -- CModel magnitude limited
AND f1.i_cmodel_mag < 20.5
AND f1.i_cmodel_mag >= 20.0

LIMIT 1000000
