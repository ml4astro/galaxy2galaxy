-- Merge forced photometry and spectroscopic sample from HSC PDR 2 wide
SELECT forced.object_id, ra, dec, tract, patch,
	-- Absorption
	a_g, a_r, a_i, a_z, a_y,
	-- Extendedness
	g_extendedness_value, r_extendedness_value, i_extendedness_value, z_extendedness_value, y_extendedness_value
	-- Fluxes
	g_cmodel_flux, g_cmodel_fluxsigma, g_cmodel_exp_flux, g_cmodel_exp_fluxsigma, g_cmodel_dev_flux, g_cmodel_dev_fluxsigma,
	r_cmodel_flux, r_cmodel_fluxsigma, r_cmodel_exp_flux, r_cmodel_exp_fluxsigma, r_cmodel_dev_flux, r_cmodel_dev_fluxsigma,
	i_cmodel_flux, i_cmodel_fluxsigma, i_cmodel_exp_flux, i_cmodel_exp_fluxsigma, i_cmodel_dev_flux, i_cmodel_dev_fluxsigma,
	z_cmodel_flux, z_cmodel_fluxsigma, z_cmodel_exp_flux, z_cmodel_exp_fluxsigma, z_cmodel_dev_flux, z_cmodel_dev_fluxsigma,
	y_cmodel_flux, y_cmodel_fluxsigma, y_cmodel_exp_flux, y_cmodel_exp_fluxsigma, y_cmodel_dev_flux, y_cmodel_dev_fluxsigma,
	-- Magnitudes
	g_cmodel_mag, g_cmodel_magsigma, g_cmodel_exp_mag, g_cmodel_exp_magsigma, g_cmodel_dev_mag, g_cmodel_dev_magsigma,
	r_cmodel_mag, r_cmodel_magsigma, r_cmodel_exp_mag, r_cmodel_exp_magsigma, r_cmodel_dev_mag, r_cmodel_dev_magsigma,
	i_cmodel_mag, i_cmodel_magsigma, i_cmodel_exp_mag, i_cmodel_exp_magsigma, i_cmodel_dev_mag, i_cmodel_dev_magsigma,
	z_cmodel_mag, z_cmodel_magsigma, z_cmodel_exp_mag, z_cmodel_exp_magsigma, z_cmodel_dev_mag, z_cmodel_dev_magsigma,
	y_cmodel_mag, y_cmodel_magsigma, y_cmodel_exp_mag, y_cmodel_exp_magsigma, y_cmodel_dev_mag, y_cmodel_dev_magsigma,
	-- Shapes
	g_sdssshape_shape11, g_sdssshape_shape12, g_sdssshape_shape22, g_sdssshape_psf_shape11, g_sdssshape_psf_shape12, g_sdssshape_psf_shape22,
	r_sdssshape_shape11, r_sdssshape_shape12, r_sdssshape_shape22, r_sdssshape_psf_shape11, r_sdssshape_psf_shape12, r_sdssshape_psf_shape22,
	i_sdssshape_shape11, i_sdssshape_shape12, i_sdssshape_shape22, i_sdssshape_psf_shape11, i_sdssshape_psf_shape12, i_sdssshape_psf_shape22,
	z_sdssshape_shape11, z_sdssshape_shape12, z_sdssshape_shape22, z_sdssshape_psf_shape11, z_sdssshape_psf_shape12, z_sdssshape_psf_shape22,
	y_sdssshape_shape11, y_sdssshape_shape12, y_sdssshape_shape22, y_sdssshape_psf_shape11, y_sdssshape_psf_shape12, y_sdssshape_psf_shape22,
	-- specz
	d_pos, d_mag, specz_ra, specz_dec, specz_redshift, specz_redshift_err, specz_mag_i

FROM pdr2_wide.forced forced
  LEFT JOIN pdr2_wide.forced2 USING (object_id)
  LEFT JOIN pdr2_wide.forced3 USING (object_id)
	INNER JOIN pdr2_wide.specz USING (object_id)
WHERE forced.isprimary AND specz.specz_flag_homogeneous

AND forced.g_inputcount_value >= 3
AND forced.r_inputcount_value >= 3
AND forced.i_inputcount_value >= 3
AND forced.z_inputcount_value >= 3
AND forced.y_inputcount_value >= 3

AND NOT forced.g_pixelflags_bright_objectcenter
AND NOT forced.r_pixelflags_bright_objectcenter
AND NOT forced.i_pixelflags_bright_objectcenter
AND NOT forced.z_pixelflags_bright_objectcenter
AND NOT forced.y_pixelflags_bright_objectcenter

AND NOT forced.g_pixelflags_bright_object
AND NOT forced.r_pixelflags_bright_object
AND NOT forced.i_pixelflags_bright_object
AND NOT forced.z_pixelflags_bright_object
AND NOT forced.y_pixelflags_bright_object

AND NOT forced.g_pixelflags_edge
AND NOT forced.r_pixelflags_edge
AND NOT forced.i_pixelflags_edge
AND NOT forced.z_pixelflags_edge
AND NOT forced.y_pixelflags_edge

AND NOT forced.g_pixelflags_saturatedcenter
AND NOT forced.r_pixelflags_saturatedcenter
AND NOT forced.i_pixelflags_saturatedcenter
AND NOT forced.z_pixelflags_saturatedcenter
AND NOT forced.y_pixelflags_saturatedcenter

AND NOT forced.g_pixelflags_interpolatedcenter
AND NOT forced.r_pixelflags_interpolatedcenter
AND NOT forced.i_pixelflags_interpolatedcenter
AND NOT forced.z_pixelflags_interpolatedcenter
AND NOT forced.y_pixelflags_interpolatedcenter

AND NOT forced.g_cmodel_flag
AND NOT forced.r_cmodel_flag
AND NOT forced.i_cmodel_flag
AND NOT forced.z_cmodel_flag
AND NOT forced.y_cmodel_flag
;
