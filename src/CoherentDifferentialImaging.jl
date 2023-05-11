module CoherentDifferentialImaging

using LinearAlgebra
# using AstroImages
using Statistics
using FFTW
using ImageTransformations
using CoordinateTransformations
using ImageFiltering
using Optim
using PSFModels


imgax(img,cx=mean(axes(img,1)), cy=mean(axes(img,2)))=axes(img,1).-cx,axes(img,2)'.-cy
imgsep(axx, axy,)= sqrt.(axx.^2 .+ axy.^2)
function circlemask(axx, axy, r, cx=mean(axx), cy=mean(axy))
	imgsep(axx,axy,) .< r
end

"""
    cdi_reconstruct(
        fringed_img,
        unfringed_image,
        mask_to_optimize;
        sidelobepos = (200.5, 229),
        sidelober = 26.5,
    )

Given fringed and unfringed self-coherent camera (SCC) images,
use the fringe information to construct a synthetic point spread
function (PSF).

An optimization process is used to reconstruct a bestfitting 
reference-beam (pinhole) PSF using low spatial frequencies.

You must provide a binary `mask_to_optimize` mask of the same size 
as the fringed and unfringed images. This selects a region of the
image to focus the optimization on.
""""""
Do the CDI.
"""
function optimalCDI(
	fringed_psf,
	unfringed_psf,
	eval_annulus;

	sidelobepos = (200.5, 229),
	sidelober = 26.5,
	subpinhole=false
)
	diff_psf = fringed_psf .- unfringed_psf
	
	fringed_psf_ft = fftshift(fft(fringed_psf))
	diff_psf_ft = fftshift(fft(diff_psf))

	sidelobe_mask =  circlemask(imgax(diff_psf_ft,sidelobepos...)..., sidelober)
	
	diff_psf_ft_sidelobe = copy(diff_psf_ft)
	diff_psf_ft_sidelobe[.! sidelobe_mask] .= 0

	fringed_psf_ft_sidelobe = copy(fringed_psf_ft)
	fringed_psf_ft_sidelobe[.! sidelobe_mask] .= 0


	# Correct for MTF
	Σs = 230
	scale = 1.0

	pixmtf_correction = let unfringed_psf=unfringed_psf
		function (Σs,scale)
			mtfmodelsinc = sinc.(
				1/(Σs) .* (
					sqrt.((axes(unfringed_psf,1)  .- mean(axes(unfringed_psf,1))).^2 .+ 
					(axes(unfringed_psf,2)' .- mean(axes(unfringed_psf,2))).^2)
				)
			)
			mtfmodelsinc[mtfmodelsinc.<=0] .= eps()
			mtfmodels_shifted = warp(
				warp(mtfmodelsinc,
				CoordinateTransformations.recenter(LinearMap(UniformScaling(scale)),mean.(axes(mtfmodelsinc))),axes(mtfmodelsinc),fillvalue=0),
					Translation(.-(sidelobepos.- mean.(axes(unfringed_psf)))),
					axes(mtfmodelsinc),
					fillvalue=0
			)
			pixel_mtf_correction = 1 ./ mtfmodelsinc .* mtfmodels_shifted
			pixel_mtf_correction[.!sidelobe_mask].=0
			pixel_mtf_correction
		end
	end

	# @info imview(sidelobe_mask .* pixmtf_correction(Σs,scale))
	
	# Construct pinhole PSF
	# pinhole_psf_ft = fill(zero(eltype(diff_psf_ft)),size(diff_psf_ft))
	# xs,ys = imgax(pinhole_psf_ft)

	# # Start the optimization by copying from the diff_psf_ft
	# m = circlemask(xs,ys,5)
	# pinhole_psf_ft[m] .= diff_psf_ft[m]
	# # pinhole_psf = abs.(ifft(pinhole_psf_ft,(1,2)))
	# # @info imview(pinhole_psf)
	
	# N = count(m)
	reconstruction_trial = let psf_ft_sidelobe = diff_psf_ft_sidelobe,
		# Allocate some scratch storage space to work with fewer allocations
		pinhole_psf_model = zeros(Float32, size(unfringed_psf)),
		reconstructed_psf = zeros(Float32, size(unfringed_psf)),
		reconstructed_psf_ft = zeros(ComplexF32, size(unfringed_psf)),
		residuals = zeros(Float32, size(unfringed_psf))
							   # pinhole_psf_ft = pinhole_psf_ft,
		pixel_mtf_correction = pixmtf_correction(Σs,scale)
		function (parameters)

			# # Construct pinhole PSF
			# pinhole_psf_ft[m] .= complex.(
			# 	view(parameters, 1:N),
			# 	view(parameters, N+1:2N),
			# )		
			# pinhole_psf = abs.(ifft(pinhole_psf_ft,(1,2)))

			x,y,fwhmx,fwhmy,theta,amp=parameters
			pinhole_psf_function = airydisk(Float32,;x,y,theta,amp,fwhm=(fwhmx,fwhmy))
			# Evaluating this analytic expression is the bottleneck
			pinhole_psf_model .= pinhole_psf_function.(
				axes(unfringed_psf,1),
				axes(unfringed_psf,2)'
			)

			# Construct pixel MTF correction
			# Σs = parameters[end-1]
			# scale = parameters[end]
			
			# pixel_mtf_correction = pixmtf_correction(Σs,scale)

			# Reconstruct
			reconstructed_psf_ft .= psf_ft_sidelobe .* pixel_mtf_correction
			ifft!(reconstructed_psf_ft)
			reconstructed_psf .= abs.(reconstructed_psf_ft).^2 ./ pinhole_psf_model
			
			if subpinhole
				reconstructed_psf .+= pinhole_psf_model
			end
			
			residuals .= unfringed_psf .- reconstructed_psf
			return std(view(residuals,eval_annulus))
		end
	end


	x0 = [
		# Parameters for the pinhole PSF model
		mean(axes(unfringed_psf,1)), # x
		mean(axes(unfringed_psf,2)), # y
		100.0, # fwhm x
		100.0, # fwhm y
		pi/2,   # theta
		0.1maximum(unfringed_psf), # amp
		
		
		# real.(diff_psf_ft[m]),
		# imag.(diff_psf_ft[m]),
		# Σs,
		# scale
	]
	
	# x0 = vcat(
	# 	real.(diff_psf_ft[m]),
	# 	imag.(diff_psf_ft[m]),
	# 	Σs,
	# 	scale
	# )
	
	result = optimize(reconstruction_trial, x0, iterations=500)#1500
	@info result

	# pinhole_psf_ft[m] .= complex.(
	# 	Optim.minimizer(result)[1:N],
	# 	Optim.minimizer(result)[N+1:2N]
	# )
	# pinhole_psf = abs.(ifft(pinhole_psf_ft,(1,2)))

	x,y,fwhmx,fwhmy,theta,amp=Optim.minimizer(result)
	pinhole_psf = airydisk(;x,y,theta,amp,fwhm=(fwhmx,fwhmy)).(axes(unfringed_psf,1),axes(unfringed_psf,2)')

	# Construct pixel MTF correction
	Σs = Optim.minimizer(result)[end-1]
	scale = Optim.minimizer(result)[end]
	# @show Σs scale
	pixel_mtf_correction = pixmtf_correction(Σs,scale)

	# Reconstruct
	reconstructed_psf = abs.(
		ifft(
			diff_psf_ft_sidelobe .* pixel_mtf_correction
			# fringed_psf_ft_sidelobe .* pixel_mtf_correction
		).^2 
	)./ pinhole_psf
	if subpinhole
		reconstructed_psf .+= pinhole_psf
	end
	

	return reconstructed_psf, pinhole_psf
end
end
